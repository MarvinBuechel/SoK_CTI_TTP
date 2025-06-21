from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserSecurityweek(ParserHTML):
    """Parser for CTI reports from securityweek.com."""

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
        title = self.get_subtag(html, 'h2', class_='page-title').get_text()
        title = self.add_punctuation(title)

        # Get content
        content = self.get_subtag(html, 'div', class_='node')
        # Remove unnecessary footer
        self.get_subtag(content, "div", class_="ad_in_content").decompose()
        self.get_subtag(content, "div", class_="sharethis").decompose()
        self.get_subtag(content, "div", class_="author_content").decompose()
        self.get_subtag(content, "div", class_="sponsored_links_box").decompose()
        self.get_subtag(content, "div", class_="author-terms").decompose()
        self.get_subtag(content, "div", id="disqus_thread").decompose()
        # Get content text
        content = content.get_text()

        # Return content as text
        return '\n\n'.join((title, content))