from typing import Iterable
from bs4 import BeautifulSoup
from bs4.element import Tag
from spacy_extensions.parsers.base import ParserHTML

class ParserDragos(ParserHTML):
    """Parser for CTI reports from dragos.com."""

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
        article = self.get_subtag(html, 'article', class_="resource")

        # Get title
        title = self.get_subtag(article, 'h1', class_='resource__title').get_text()

        # Add punctuation
        title = self.add_punctuation(title)

        # Get content
        content = self.get_subtag(article, 'div', class_="resource__main")

        # Remove meta
        meta = self.get_subtag(content, 'div', class_="resource-meta")
        meta.decompose()

        # Traverse content to get tables
        for child in content.descendants:
            if child.name == 'table':
                child.string = self.parse_table(child)

        # Get content text
        content = content.get_text()

        # Return result
        return '\n\n'.join((title, content))
