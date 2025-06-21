from scholarly import scholarly, MaxTriesExceededException
from tqdm import tqdm
import pandas as pd
import requests
import time
import random

from datetime import datetime
import os
import logging as logger
from urllib.parse import quote

from scholarly import scholarly, ProxyGenerator


WAIT_TIME = 5
SCRAPED_DATA_FOLDER = "./data"
os.makedirs(SCRAPED_DATA_FOLDER, exist_ok=True)


proxies = {
  "http": ""
}

logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCHOLAR_KEYWORDS = [
    "cyber threat intelligence mining",
    "cyber threat intelligence extraction",
    "CTI mining",
    "CTI extraction",
    "TTP extraction",
    "TTP mining",
    "att&ck extraction",
    "att&ck mining",
    "attack pattern extraction",
    "attack pattern mining",
    "attack technique extraction",
    "attack technique mining",
    "attack graph extraction",
    "attack graph mining"
]

DBLP_KEYWORDS = [
    "threat$ intelligence|action extraction|mining|classification",
    "cti$ extraction|mining|classification",
    "TTP extraction|mining|classification",
    "att&ck extraction|mining|classification",
    "attack pattern|technique|behavior|graph extraction|mining|classification"
    "threat report extraction|mining|classification"
]

class TooManyRequests(Exception):
    pass

def sleep_with_jitter(base=WAIT_TIME):
    time.sleep(base + random.uniform(0, 3))

def scrape_google_scholar():
    for kw in tqdm(SCHOLAR_KEYWORDS):
        logger.info(f"Keyword: {kw}")
        outfile = os.path.join(SCRAPED_DATA_FOLDER, f"{kw}_scholar.csv")
        if os.path.basename(outfile) in os.listdir(SCRAPED_DATA_FOLDER):
            logger.info(f"Keyword {kw} already scraped!")
            continue

        kw_data = []
        i = 0
        max_results = 100  # Approx. 10 pages

        try:
            search_query = scholarly.search_pubs(kw)
            for entry in search_query:
                if i >= max_results:
                    break

                bib = entry.get("bib", {})
                kw_data.append({
                    "authors": ", ".join(bib.get("author", [])),
                    "title": bib.get("title"),
                    "venue": bib.get("venue"),
                    "year": bib.get("pub_year"),
                    "url": entry.get("pub_url"),
                    "doi": None,
                    "type": None,
                    "citations": entry.get("num_citations"),
                })
                i += 1
                print(f"Papers found: {i}", end="\r")
                if i % 10 == 0:
                    sleep_with_jitter()

        except MaxTriesExceededException:
            logger.error(f"Max tries exceeded for keyword: {kw}")
        except Exception as e:
            logger.error(f"{kw}: {type(e).__name__} at {i} papers", exc_info=True)

        kw_data_df = pd.DataFrame(kw_data)
        kw_data_df.to_csv(outfile, index=False)


def get_with_retries(url, max_retries=5):
    delay = WAIT_TIME
    for attempt in range(max_retries):
        res = requests.get(url, proxies=proxies, verify=False)
        if res.status_code == 429:
            logger.warning(f"429 received. Retrying in {delay} seconds...")
            time.sleep(delay + random.uniform(0, 2))
            delay *= 2  # Exponential backoff
        else:
            return res
    raise TooManyRequests("Exceeded retry attempts due to rate limiting.")


def scrape_dblp():


    for kw in tqdm(DBLP_KEYWORDS):
        kw = quote(kw)
        logger.info(f"Keyword: {kw}")
        outfile = os.path.join(SCRAPED_DATA_FOLDER, f"{kw}_dblp.csv")
        if os.path.basename(outfile) in os.listdir(SCRAPED_DATA_FOLDER):
            logger.info(f"Keyword {kw} already scraped!")
            continue

        kw_data = []
        i = 0
        f = 0
        is_empty = False

        try:
            while not is_empty:
                url = f"https://dblp.org/search/publ/api?q={kw}&format=json&h=1000&f={f}"
                res = get_with_retries(url)

                search_results = res.json()["result"]["hits"].get("hit", [])
                f += 1000
                if len(search_results) == 0:
                    is_empty = True

                for entry in search_results:
                    info = entry["info"]
                    authors = info.get("authors", {}).get("author", {"text": None})
                    if isinstance(authors, dict):
                        author_txt = authors["text"]
                    else:
                        author_txt = ", ".join([a["text"] for a in authors])

                    kw_data.append({
                        "authors": author_txt,
                        "title": info.get("title"),
                        "venue": info.get("venue"),
                        "year": info.get("year"),
                        "url": info.get("ee"),
                        "doi": info.get("doi"),
                        "type": info.get("type"),
                        "citations": None,
                    })
                    i += 1
                    print(f"Papers found: {i}", end="\r")

                sleep_with_jitter()

        except TooManyRequests:
            logger.error(f"Too many requests for keyword {kw}. Skipping...")
        except Exception as e:
            logger.error(f"{kw}: {type(e).__name__} at {i} papers", exc_info=True)

        kw_data_df = pd.DataFrame(kw_data)
        kw_data_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    scrape_google_scholar()
    scrape_dblp()

