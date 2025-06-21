from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserMcafee(ParserHTML):
    """Parser for CTI reports from mcafee.com."""

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
        title = self.get_subtag(html, "h1", class_="main-heading").get_text()
        title = self.add_punctuation(title)

        # Get content
        content = self.get_subtag(html, "div", class_="the_content").get_text()

        # Return text
        return '\n\n'.join((title, content))