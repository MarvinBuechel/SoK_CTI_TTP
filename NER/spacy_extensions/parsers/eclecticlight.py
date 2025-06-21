from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserEclecticlight(ParserHTML):
    """Parser for CTI reports from eclecticlight.co."""

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
        title = self.get_subtag(html, "h1", class_="entry-title").get_text()
        title = self.add_punctuation(title)

        # Get content
        content = self.get_subtag(html, "div", class_="entry-content")
        # Remove footer
        self.get_subtag(content, "div", id="jp-post-flair").decompose()
        # Get content text
        content = content.get_text()

        # Return result
        return '\n\n'.join((title, content))
