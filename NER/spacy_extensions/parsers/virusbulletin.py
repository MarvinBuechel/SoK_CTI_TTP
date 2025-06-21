from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from spacy_extensions.parsers.base import ParserHTML
from typing import Iterator

class ParserVirusbulletin(ParserHTML):
    """Parser for CTI reports from virusbulletin.com."""

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
        # Get titlepage
        main = self.get_subtag(html, "div", class_="col-md-9")

        # Return text
        return "\n".join(self.traverse_virusbulletin(main))

    
    def traverse_virusbulletin(self, html: Tag) -> Iterator[str]:
        """Iterate over text in virusbulletin HTML pages."""
        if isinstance(html, NavigableString):
            yield str(html.string)
        elif html.name.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            yield '\n' + html.get_text()
        elif html.name.lower() in {"p"}:
            yield html.get_text()
        elif "ccm-remo-expand" in html.get("class", []):
            yield ""
        else:
            for child in html.children:
                yield from self.traverse_virusbulletin(child)