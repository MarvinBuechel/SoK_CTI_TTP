from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from typing import Iterable, Iterator, Optional
import re

class ParserHTML:

    def parse(self, html: str) -> str:
        """Parse an HTML file to extract text.
        
            Parameters
            ----------
            html : str
                HTML from which to extract text.
                
            Returns
            -------
            text : str
                Content of HTML file.
            """
        # Transform HTML into beautifulsoup object
        soup = BeautifulSoup(html, 'html.parser')

        # Parse soup - specific for subparser
        text = self.parse_specific(soup)

        # Replace multiple whitespace lines
        regex_whitespace = re.compile(r'\n\s*\n', )
        text = regex_whitespace.sub('\n\n', text).strip() + '\n'

        # Strip all lines
        text = '\n'.join(line.strip() for line in text.split('\n'))

        # Return result
        return text

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
        raise NotImplementedError("Subparser should implement parse_specific.")

    ########################################################################
    #                            Shared methods                            #
    ########################################################################

    def get_subtag(self, tag: Tag, name: str, *args, **kwargs):
        """Retrieve a subtag from a tag.
            
            Parameters
            ----------
            tag : Tag
                Element in which to search for subtag.
                
            name : str
                Name of subtag, e.g., div
            
            *args : Any
                Args passed to Beautifulsoup.find_all() method.
                
            **kwargs : Any
                Kwargs passed to Beautifulsoup.find_all() method.
            
            
            Returns
            -------
            subtag : Tag
                Found subtag.
            """
        # Ret resulting subtag(s)
        result = tag.find_all(name, *args, **kwargs)
        # Assert we only found a single subtag
        assert len(result) == 1, (
            f"{len(result)} elements found for <{name}> [{args}, {kwargs}], "
            "expected 1."
        )
        # Return result
        return result[0]
    
    def add_punctuation(
            self,
            string    : str,
            end       : str = '.',
            exceptions: Iterable[str] = ',.?!:;'
        ) -> str:
        """Add punctuation to string if it does not end in punctuation.
            
            Parameters
            ----------
            string : str
                String for which to add punctuation.
                
            end : str, default='.'
                Punctuation to end string with.

            exceptions : Iterable[str], default = ',.?!:;'
                If string ends in one of the given strings, do not add
                punctuation.
            """
        # Add punctuation if necessary
        if not any(string.endswith(exception) for exception in exceptions):
            string = string.rstrip() + end

        # Return string
        return string

    def get_text(
            self,
            html: Tag,
            add_punctuation:bool = True,
        ) -> str:
        """Extract text from beautifulsoup tag. Normally this is done using the
            get_text() method of the beautifulsoup tag. However, this does not
            deal with newlines properly. Furthermore, websites often do not add
            proper punctuation to the end of their sentences. This can be added
            with the get_text method.
            
            Parameters
            ----------
            html : Tag
                BeautifulSoup tag from which to extract text.
            
            add_punctuation : bool, default=False
                If True, add punctuation to end of sentences.
            """
        # Initialise result
        result = list()

        # Get all text in tag
        for tag, text in traverse(html):
            # Skip non-existing text
            if not text.strip(): continue

            # Add punctuation if necessary
            if add_punctuation:
                text = self.add_punctuation(text)
            
            # Add to result
            result.append(text)
        
        # Return result
        return '\n\n'.join(result)

    ########################################################################
    #                         Table parsing method                         #
    ########################################################################

    def parse_table(self, table: Tag, row_separator='\n', cell_separator='\t'):
        """Get text from table tag.
            
            Parameters
            ----------
            table : Tag
                Table from which to retrieve text.
            
            Returns
            -------
            text : str
                Text in table.
            """

        # Get table header
        header = self.parse_table_rows(
            table.find_all('thead'),
            row_separator  = row_separator,
            cell_separator = cell_separator,
        )

        # Get table body
        body = self.parse_table_rows(
            self.get_subtag(table, 'tbody').find_all('tr'),
            row_separator  = row_separator,
            cell_separator = cell_separator,
        )

        # Return result
        return '\n'.join((header, body))
        
        
    def parse_table_rows(
        self,
        rows: Iterable[Tag],
        row_separator : str = '\n',
        cell_separator: str = '\t',
    ):
        """Parse table rows from a list of row tags.
            
            Parameters
            ----------
            rows : Iterable[Tag]
                Rows to parse.

            row_separator : str, default='\n'
                Separator to use for rows.

            cell_separator : str, default='\t'
                Separator to use for cells.
            """
        # Iterate over all rows
        for index, row in enumerate(rows):
            # Get text from rows
            rows[index] = cell_separator.join(
                cell.get_text() for cell in row.find_all('td')
            )

        # Join and return rows
        return row_separator.join(rows)

def traverse(html: Tag) -> Iterator[str]:
    """Traverse all strings in BeautifulSoup Tag.
    
        Parameters
        ----------
        html : Tag
            Component to traverse.
            
        Yields
        ------
        text : str
            Text in Tags sub-components
        """
    # Loop over all children in Tag
    for child in html.children:
        if isinstance(child, Tag):
            yield from traverse(child)
        elif isinstance(child, NavigableString):
            yield (html.name, child.text)