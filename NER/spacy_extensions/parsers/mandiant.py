from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserMandiant(ParserHTML):
    """Parser for CTI reports from mandiant.com."""

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
        title = self.add_punctuation(self.get_subtag(html, 'h1').get_text())

        # Get content
        content = self.get_subtag(html, "div", class_="resource-body").get_text()

        # return result
        return '\n\n'.join((title, content))