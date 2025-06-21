from bs4 import BeautifulSoup
from bs4.element import Tag
from spacy_extensions.parsers.base import ParserHTML

class ParserWelivesecurity(ParserHTML):
    """Parser for CTI reports from welivesecurity.com."""

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
        # Get promo content text
        text = [self.get_subtag(html, "div", class_ = "promo-text").get_text()]

        # Get article content
        article = self.get_subtag(html, "div", class_="col-md-10 col-sm-10 col-xs-12 formatted")
         
        # Iterate over children
        for child in article.children:
            # Break on dot tag
            if isinstance(child, Tag):
                # Check if child contains any class with dot tag
                if 'dot' in child.get('class', []) or len(child.find_all('div', class_='dot')):
                    break
                # Check if child contains any class with widgets tag
                if 'widgets' in child.get('class', []) or len(child.find_all('div', class_='widgets')):
                    break
            
            # Add text contents
            text.append(child.get_text())
        else:
            raise ValueError("Could not find <dot> tag")

        # Return text
        return '\n'.join(text)