from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserForcepoint(ParserHTML):
    """Parser for CTI reports from blogs.forcepoint.com."""

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
        # Initialise result
        result = list()

        # Get title
        title = html.find_all("h1")
        if len(title) != 1:
            print(f"Warning: Multiple titles found: '{len(title)}': {title}")
        result.append(title[0].get_text())

        # Get summary
        summary = html.find_all("div", class_="pane-node-field-blog-summary")
        assert len(summary) <= 1, f"Multiple summaries found: '{len(summary)}'"
        if len(summary):
            result.append(summary[0].get_text())


        content = html.find_all("div", class_="pane-node-field-main-content")
        assert len(content) == 1, f"Multiple contents found: '{len(content)}'"
        result.append(content[0].get_text())

        # Return text
        return '\n'.join(result)