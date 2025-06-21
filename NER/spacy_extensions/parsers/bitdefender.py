from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserBitdefender(ParserHTML):
    """Parser for CTI reports from bitdefender.com."""

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
        # Get title
        title = self.add_punctuation(html.title.get_text())

        # Get article
        article = self.get_subtag(html, "div", class_="single-post__content").get_text()

        # Return text
        return '\n\n'.join((title, article))