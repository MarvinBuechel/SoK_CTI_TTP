from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserFireeye(ParserHTML):
    """Parser for CTI reports from fireeye.com."""

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
        title = html.title.get_text()

        # Get article - entrytext
        try:
            article = self.get_subtag(html, 'div', class_="entrytext").get_text()
        # If not available, get article - resource-body
        except AssertionError:
            article = self.get_subtag(html, 'div', class_="resource-body").get_text()

        # Add punctuation
        title = self.add_punctuation(title)

        # Return result
        return '\n\n'.join((title, article))
        