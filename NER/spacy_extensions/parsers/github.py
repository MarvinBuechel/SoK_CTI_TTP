from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserGithub(ParserHTML):
    """Parser for CTI reports from github.com."""

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
        return self.get_subtag(html, 'article', class_="entry-content").get_text()
