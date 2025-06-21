from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserPaloaltonetworks(ParserHTML):
    """Parser for CTI reports from paloaltonetworks.com."""

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

        # Get article
        article = self.get_subtag(html, "div", class_="article__content")
        # Remove footer
        self.get_subtag(article, 'div', class_='article__subscribe').decompose()

        # Return text
        return '\n\n'.join((title, article.get_text()))