from bs4 import BeautifulSoup
from spacy_extensions.parsers.base import ParserHTML

class ParserTrendmicro(ParserHTML):
    """Parser for CTI reports from trendmicro.com."""

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
        # Trendmicro has different ways of presenting articles. Some articles 
        # are presented within the <article> tag, others within the <section>
        # tag. Therefore, we present two methods for parsing trendmicro
        try:
            article = self.get_subtag(html, "article")
            setup = "article"
        except AssertionError:
            setup = "section"

        if setup == "article":
            return self.parse_specific_article(html)
        elif setup == "section":
            return self.parse_specific_section(html)


    def parse_specific_article(self, html: BeautifulSoup) -> str:
        """Parse beautifulsoup of trendmicro in article mode
        
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
        article = self.get_subtag(html, "article")
        # Get title
        title = self.get_subtag(article, "h1", class_="article-details__title")
        # Get description
        description = self.get_subtag(article, "p", class_="article-details__description")

        # Get main
        main = self.get_subtag(article, "main")
        # Remove tags
        for element in main.find_all("section", class_="tag--list"):
            element.decompose()

        # Create output
        title       = self.add_punctuation(title.get_text())
        description = self.add_punctuation(description.get_text())
        main = main.get_text()

        # Return result
        return '\n\n'.join((title, description, main))


    def parse_specific_section(self, html: BeautifulSoup) -> str:
        """Parse beautifulsoup of trendmicro in section mode
        
            Parameters
            ----------
            html : BeautifulSoup
                Soup to parse.
                
            Returns
            -------
            text : str
                Content of HTML file.
            """
        # Get header and content
        header  = self.get_subtag(html, "section", class_="articleHeader")
        content = self.get_subtag(html, "section", class_="articleContent")

        # Get title
        title = self.add_punctuation(self.get_subtag(header, "h1").get_text())

        # Remove footer
        self.get_subtag(content, "div", class_="postedIn").decompose()
        # Get content text
        content = content.get_text()
        
        # Return result
        return '\n\n'.join((title, content))