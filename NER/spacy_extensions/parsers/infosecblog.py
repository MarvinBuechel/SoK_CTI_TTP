from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserInfosecblog(ParserHTML):
    """Parser for CTI reports from infosecblog.org."""

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
        article = self.get_subtag(html, "article", class_="post")

        # Remove footer from article
        article.footer.decompose()

        # Return text
        return article.get_text()