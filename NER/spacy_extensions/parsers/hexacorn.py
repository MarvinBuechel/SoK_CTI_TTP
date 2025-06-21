from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserHexacorn(ParserHTML):
    """Parser for CTI reports from hexacorn.com."""

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
        # Get content
        return self.get_subtag(html, 'div', class_="post-content").get_text()
    