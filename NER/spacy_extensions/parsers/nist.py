from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserNist(ParserHTML):
    """Parser for CTI reports from nvd.nist.gov."""

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
        return self.get_subtag(html, 'div', id='vulnDetailPanel').get_text()