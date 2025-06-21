import pandas as pd
import re



def replace_iocs(text):
    # Regular expression patterns for different types of IoCs
    url_pattern = re.compile(r'\(?http(s)?://[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    url_pattern_masked = re.compile(r'\(?(http|hxxp)(s)?:\/\/[-a-zA-Z0-9@:%._\+~#=]{2,256}(\[)?\.(\])?[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)')

    ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\[?\.\]?){3}[0-9]{1,3}\b')

    hash_pattern = re.compile(r'\b[A-Fa-f0-9]{32}\b|\b[A-Fa-f0-9]{40}\b|\b[A-Fa-f0-9]{64}\b')

    # Replace URLs with [URL]
    text = url_pattern.sub('[URL]', text)
    text = url_pattern_masked.sub('[OBFUSCATED URL]', text)
    # Replace IP addresses with [IP]
    text = ip_pattern.sub('[IP]', text)
    # Replace hashes with [HASH]
    text = hash_pattern.sub('[HASH]', text)

    return text


def convert_mitrival():
    df = pd.read_json(
        "D:\Datasets\cyber_threat_intelligence\dataset\MITREtrieval\original_fixed\MITREtrievel700cti_ds.json")

    for index, row in df.iterrows():
        cti_report = row["cti_report"]
        mitre_ids = row["mitre_ids"]  # not unique ids, id in order to appearance in text

        cleaned_cti_report = replace_iocs(cti_report)

        row["cti_report"] = cleaned_cti_report

    df.to_json("D:\Datasets\cyber_threat_intelligence\dataset\MITREtrieval\ip_hash_cleaned\MITREtrievel700cti_ds.json")
