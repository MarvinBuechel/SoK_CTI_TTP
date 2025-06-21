from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserSecurity(ParserHTML):
    """Parser for CTI reports from security.com."""

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
        article = self.get_subtag(html, 'article')
        # Get content
        content = self.get_subtag(article, 'div', class_="blog-post__content")
        # Get header
        header = content.find_all('header')[0]

        # Get title
        title    = self.add_punctuation(self.get_subtag(header, 'h1').get_text())
        subtitle = self.add_punctuation(self.get_subtag(header, 'h2').get_text())

        # Remove header
        header.decompose()

        # Get content
        content = content.get_text()

        # return result
        return '\n\n'.join((title, subtitle, content))