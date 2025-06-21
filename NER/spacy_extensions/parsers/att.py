from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserAtt(ParserHTML):
    """Parser for CTI reports from att.com."""

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
        title = self.get_subtag(html, "div", class_="blog-title-date-author-area")
        title = self.add_punctuation(self.get_subtag(title, "h1").get_text())

        # Get content
        content = self.get_subtag(html, "div", class_="blog-body").get_text()

        # Return text
        return '\n\n'.join((title, content))