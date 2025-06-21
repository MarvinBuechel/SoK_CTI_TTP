from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserLolbas(ParserHTML):
    """Parser for CTI reports from lolbas-project.github.io."""

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
        # Return entire page text
        return html.get_text()
