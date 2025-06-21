from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserCybereason(ParserHTML):
    """Parser for CTI reports from cybereason.com."""

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
        article = self.get_subtag(html, 'div', class_="cr-mln__blog-post")

        # Get title
        title = self.get_subtag(article, 'h1').get_text()
        title = self.add_punctuation(title)

        # Get content
        content = self.get_subtag(
            article,
            'span',
            id='hs_cos_wrapper_post_body'
        ).get_text()

        # Return result
        return '\n\n'.join((title, content))
