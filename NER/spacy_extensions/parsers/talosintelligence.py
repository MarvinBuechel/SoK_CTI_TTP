from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserTalosintelligence(ParserHTML):
    """Parser for CTI reports from talosintelligence.com."""

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

        # Remove unnecessary content
        self.get_subtag(post, "div", class_="post-footer").decompose()        

        # Return text
        return post.get_text()