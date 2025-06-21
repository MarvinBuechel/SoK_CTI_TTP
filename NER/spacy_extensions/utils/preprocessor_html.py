from bs4 import BeautifulSoup, NavigableString, Comment
import re

class PreprocessorHTML(object):

    def __init__(self):
        """Preprocessor for preprocessing text such that sentences are complete.
            """
        # Regexes
        self.REGEX_WHITESPACE = re.compile(r'\s+')

        # Store spans
        self.spans = dict()

    ########################################################################
    #                              Preprocess                              #
    ########################################################################

    def html(self, string, include_title=True):
        """Preprocess HTML text such that SpaCy can deal with the text as proper
            sentences.

            Parameters
            ----------
            string : string
                HTML text as string.

            Returns
            -------
            parsed : string
                Parsed string that SpaCy can deal with as text.

            spans : dict
                Dictionary of character spans retrieved from HTML.
            """
        # Initialise result
        result = ""

        # Reset spans
        self.spans = dict()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(string, 'html.parser')

        ####################################################################
        #                          Include title                           #
        ####################################################################

        # Start result with title
        if include_title and soup.title:
            # Extract title
            title = soup.title.get_text()
            # Close title with period if required
            if not title.endswith('.'):
                title = f"{title}."

            # Add title
            result += f"{title}\n"

        ####################################################################
        #                            Parse text                            #
        ####################################################################

        # Parse HTML data
        result = self.feed(soup, result=result)

        ####################################################################
        #                          Return result                           #
        ####################################################################

        # Return result
        return result, self.spans


    ########################################################################
    #                     Recursively parse HTML data                      #
    ########################################################################

    def feed(self, html, result=None):
        """Recursively parse HTML data.

            Parameters
            ----------
            html : bs4.BeautifulSoup
                Soup HTML to parse.

            result : string, optional
                If given, use the resulting string as start input.

            Returns
            -------
            result : string
                Parsed HTML text.
            """
        # Initialise result, if required
        if result is None:
            result = ""

        # Handle start tag of element
        result, ignore = self.handle_start(result, html)

        # If ignore tag, return result
        if ignore: return result

        # Loop over all HTML children elements
        if hasattr(html, 'children'):
            for element in html.children:
                result = self.feed(element, result)

        # Handle end tag of element
        result = self.handle_end(result, html)

        # Return result
        return result


    def handle_start(self, result, element):
        """Handle the start of an element.
            Allows us to add data depending on the tag of the element.

            Parameters
            ----------
            result : string
                Result text for which to append data.

            element : bs4.BeautifulSoup
                BeautifulSoup element for which to handle the start.
            """
        # Ignore tags
        if (
                isinstance(element, Comment) or
                element.name in {'head', 'meta', 'style', 'script'}
            ):
            return result, True

        # String tags
        elif isinstance(element, NavigableString) and element.parent.name != '[document]':
            # Get string of element
            string = element.string

            # In HTML, all whitespace is a single space
            # Replace whitespace with space
            if element.parent.name != 'pre':
                string = re.sub(self.REGEX_WHITESPACE, ' ', string)

            # Strip string
            string = string.strip()

            # Check if we have anything to add
            if string:

                # If not on newline, add space
                if not result.endswith('\n'):
                    result += ' '

                # Add string to result
                result += f"{string} "

        # Header tags
        elif element.name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            result += '\n'

        # Emphasis
        elif element.name in {'b', 'i', 'emph'}:
            result += '"'

        # Code
        elif element.name in {'code', 'pre'}:
            if 'code' not in self.spans:
                self.spans['code'] = {
                    'stack': list(),
                    'spans': list(),
                }
            self.spans['code']['stack'].append(len(result))

        # Start paragraph with newline
        elif element.name == "p":
            result += '\n'

        # List tags
        elif element.name == 'li':
            result += '    '

        # Return result
        return result, False


    def handle_end(self, result, element):
        """Handle the end of an element.
            Allows us to add data depending on the tag of the element.

            Parameters
            ----------
            result : string
                Result text for which to append data.

            element : bs4.BeautifulSoup
                BeautifulSoup element for which to handle the end.
            """
        # End with punctuation
        if element.name in {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'}:
            # End header tag with punctuation if necessary
            if result.strip()[-1] not in '.,;:!?()"\'`':
                # Special case for list
                if element.name == 'li':
                    result = result.strip() + ';'

                # Otherwise, add period
                else:
                    result = result.strip() + '.'

        # End paragraph with newline
        if element.name == "p":
            result += '\n'

        # Emphasis
        elif element.name in {'b', 'i', 'emph'}:
            result += '"'

        # Code
        elif element.name in {'code', 'pre'}:
            if 'code' not in self.spans:
                self.spans['code'] = {
                    'stack': list(),
                    'spans': list(),
                }
            self.spans['code']['spans'].append((
                self.spans['code']['stack'].pop(),
                len(result)
            ))

            if element.name == 'pre':
                result += '\n'

        # List items
        elif element.name == 'li':
            result += '\n'

        # Close list with period
        elif element.name in {'ol', 'ul'}:
            if result.strip().endswith(';'):
                result = result.strip()[:-1] + '.\n'

        # Return result
        return result
