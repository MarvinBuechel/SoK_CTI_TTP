from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserMedium(ParserHTML):
    """Parser for CTI reports from medium.com."""

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
        # Get article
        article = self.get_subtag(html, "article")
        # Get actual content
        content = self.get_subtag(article, 'section')
        
        # Return result
        return content.get_text()