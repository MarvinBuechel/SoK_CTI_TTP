from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserOffice(ParserHTML):
    """Parser for CTI reports from support.office.com."""

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
        # Get content
        content = self.get_subtag(html, 'div', id="supArticleContent")
        # Get title
        title = self.get_subtag(content, 'h1', id='page-header')
        # Get main content
        main = self.get_subtag(content, 'article', class_='ocpArticleContent')

        # Prepare result
        title = self.add_punctuation(title.get_text())
        main  = main.get_text()

        # Return content as text
        return '\n\n'.join((title, main))