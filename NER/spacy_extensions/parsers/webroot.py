from bs4 import BeautifulSoup
from bs4.element import Tag
from spacy_extensions.parsers.base import ParserHTML

class ParserWebroot(ParserHTML):
    """Parser for CTI reports from webroot.com."""

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
        title = self.get_subtag(html, "h1", class_="entry-title")
        # Get content
        content = self.get_subtag(html, "div", class_="entry-content")
        children = list(filter(lambda x: isinstance(x, Tag), content.children))
        assert len(children) == 1, f"Multiple children found: {len(children)}"
        content = children[0]

        # Remove bio and social
        to_remove = list()
        for child in content.find_all('div', id='singleBios'):
            to_remove.append(child)
        assert len(to_remove) == 1, f"Could not remove single bio: {len(to_remove)}"

        for child in content.find_all('div', class_="et_social_inline"):
            to_remove.append(child)
        assert len(to_remove) == 3, f"Could not remove single bio: {len(to_remove)}"

        for child in to_remove:
            child.decompose()

        # Return content text
        return content.get_text()