from bs4 import BeautifulSoup
from bs4.element import Tag
from spacy_extensions.parsers.base import ParserHTML

class ParserSophos(ParserHTML):
    """Parser for CTI reports from sophos.com."""

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
        # Retrieve main
        main = self.get_subtag(html, "main", id="main")

        # Get article childs of main
        article = None
        for child in main.children:
            if isinstance(child, Tag) and child.name == "article":
                assert article is None, "Multiple articles found"
                article = child

        # Return text
        return article.get_text()