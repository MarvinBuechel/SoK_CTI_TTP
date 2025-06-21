from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserTwitter(ParserHTML):
    """Parser for CTI reports from twitter.com."""

    def parse_specific(self, html: BeautifulSoup) -> str:
        """Parse beautifulsoup, should be implemented by subparsers.
        
            Parameters
            ----------
            html : BeautifulSoup
                Soup to parse.
                
            Returns
            -------
            text : str
                Content of HTML file.
            """
        # Get tweets
        tweets = html.find_all('div', **{"data-testid":"tweetText"})

        # Initialise result
        tweet_texts = list()

        for tweet in tweets:
            # Get tweet text
            tweet_texts.append(self.add_punctuation(tweet.get_text()))
        
        # Return tweets in thread
        return '\n\n'.join(tweet_texts)
