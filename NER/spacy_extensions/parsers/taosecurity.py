from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserTaosecurity(ParserHTML):
    """Parser for CTI reports from taosecurity.blogspot.com."""

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
        # Get post
        post = self.get_subtag(html, "div", class_="post")

        # Remove post bottom
        to_remove = list()
        for bottom in post.find_all('div', class_="post-bottom"):
            to_remove.append(bottom)
        assert len(to_remove) == 1, "Multiple items to remove"
        for child in to_remove:
            child.decompose()

        # Return text
        return post.get_text()