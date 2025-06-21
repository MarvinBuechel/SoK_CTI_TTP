from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserSucuri(ParserHTML):
    """Parser for CTI reports from sucuri.net."""

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
        # Get title content
        titles   = html.find_all("h1" , class_ = "entry-title")
        contents = html.find_all("div", class_ = "entry-content")

        # Check for same number of titles and contents
        assert len(titles) == len(contents), (
            f"Different number of titles ('{len(titles)}') and contents "
            f"('{len(contents)}') found."
        )

        if len(titles) != 1:
            print(f"Warning: {len(titles)} posts found in single page.")

        # Special case: https://blog.sucuri.net/2019/01/owasp-top-10-security-risks-part-iv.html
        if len(titles) == 0:
            # Get titles
            titles   = html.find_all("h1", class_="mb-30 green")
            contents = html.find_all("div", id="step0")

            # Assert titles and contents are of length 1
            assert len(titles) == len(contents) == 1, "Special case failed."

            # Loop over children
            to_remove = list()
            for child in contents[0].find_all('div', class_='c-lg-12'):
                if 'step' not in child.get('id', ''):
                    to_remove.append(child)
                    
            for child in to_remove:
                child.decompose()
            

        # Initialise result
        result = list()

        # Loop over titles and contents
        for title, content in zip(titles, contents):
            
            # Remove aside
            for aside in content.find_all('aside'):
                aside.decompose()

            result.append(title.get_text() + '\n' + content.get_text())

        # Return text
        return '\n\n'.join(result)