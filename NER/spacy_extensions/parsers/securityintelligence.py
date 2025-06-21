from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserSecurityintelligence(ParserHTML):
    """Parser for CTI reports from securityintelligence.com."""

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
        title = self.add_punctuation(html.title.get_text())

        # Get content
        content = self.get_subtag(html, "main", id="post__content")
        # Remove tags and author
        self.get_subtag(content, 'div', class_="post__tags").decompose()
        self.get_subtag(content, 'div', class_="post__author").decompose()
        # Get content text
        content = content.get_text()

        # Return result
        return '\n\n'.join((title, content))