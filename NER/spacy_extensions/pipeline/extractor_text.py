# Imports
import magic
import os
import re
import spacy
import subprocess
import tempfile
import warnings
import zipfile
from spacy_extensions.utils import SpacyCallable
from typing import Literal, Optional

# Local imports
from spacy_extensions.utils import PreprocessorHTML

@spacy.Language.factory('extractor_text')
class ExtractorText(SpacyCallable):
    """ExtractorText for extracting text from various documents.
        The ExtractorText allows you to convert additional file types to
        documents in the SpaCy NLP pipeline."""

    def __init__(
            self,
            nlp,
            name,
            pdfact        : Optional[str] = None,
            default       : Literal['error', 'pdf', 'html', 'ascii', 'zip'] = 'error',
            always_default: bool          = False,
            spans         : bool          = True,
        ):
        """ExtractorText for extracting text from various documents.
            The ExtractorText allows you to convert additional file types to
            documents in the SpaCy NLP pipeline.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which ExtractorText is used.

            name : string
                Name of pipeline.

            Config Parameters
            -----------------
            pdfact : Optional[str], default=None
                Path to .jar file for pdfact, see
                `https://github.com/ad-freiburg/pdfact`.

            default : Literal['error', 'pdf', 'html', 'ascii', 'zip'], default='error'
                Default extraction technique to use if filetype could not be
                found automatically. If 'error', do not use any extraction, but
                throw an error instead.

            always_default : bool, default=False
                If True, always use the default parsing technique, do not look
                for automatic techniques.

            spans : bool, default=True
                If True, include spans extracted from HTML.

            Note
            ----
            ExtractorText will override the ``_ensure_doc`` method of your
            SpaCy NLP Language (the nlp object). This may have consequences if
            you modify the ``_ensure_doc`` method in other code. If so, please
            add the ExtractorText to your pipeline **after** your original
            modification (by creating the ExtractorText later, **not** by using
            the after parameter in ``add_pipe('text_extractor', after=''))``.
            If you are required to first initiate the ExtractorText before
            modifying the ``_ensure_doc`` method, please modify the
            ``ExtractorText._ensure_doc_original`` method instead. This method
            will be called after extracting the text from the original document.
            """
        # Set SpaCy pipeline objects
        self.nlp  = nlp
        self.name = name

        # Override original _ensure_doc method
        self._ensure_doc_original = self.nlp._ensure_doc
        self.nlp._ensure_doc      = self.convert_document

        # Set HTML preprocessor
        self.preprocessor = PreprocessorHTML()

        # Set configuration
        self.pdfact         = pdfact
        self.default        = default
        self.always_default = always_default
        self.spans          = spans

        # Set regex for paragraph detection
        self.regex_paragraphs = re.compile(r'\n\s*\n')

    ########################################################################
    #                       SpaCy default extraction                       #
    ########################################################################

    def convert_document(self, document):
        """Extract text from any supported document type and convert it to a Doc
            object.

            Parameters
            ----------
            document : spacy.tokens.Doc | string | bytes
                Document to convert.
                Only modifies the original Language `_ensure_doc` method if
                given document is of type bytes. Otherwise, will invoke the
                original method.

            Returns
            -------
            document : spacy.tokens.Doc
                SpaCy Doc object.
            """
        # Initialise spans
        spans = None

        # If document is not yet parsed, parse document
        if isinstance(document, bytes):
            # Guess file type
            file_type = magic.from_buffer(document)

            # Parse document based on file_type
            if not self.always_default and 'PDF' in file_type:
                document = self.pdf(document)
            elif not self.always_default and 'HTML' in file_type:
                document, spans = self.html(document)
            elif not self.always_default and 'ASCII' in file_type or file_type.startswith('UTF-8 Unicode text'):
                document = self.ascii(document)
            elif not self.always_default and 'Zip' in file_type:
                document = self.zip(document)

            # If file type is not supported, fallback on default
            elif self.default is not None and self.default.lower() == 'pdf':
                document = self.pdf(document)
            elif self.default is not None and self.default.lower() == 'html':
                document, spans = self.html(document)
            elif self.default is not None and self.default.lower() == 'ascii':
                document = self.ascii(document)
            elif self.default is not None and self.default.lower() == 'zip':
                document = self.zip(document)

            # If no default type is given, raise error
            else:
                raise ValueError(f"Unsupported file type '{file_type}'.")

        # Convert to document using original method
        result = self._ensure_doc_original(document)

        ####################################################################
        #                            Set spans                             #
        ####################################################################

        # Add spans to document
        if self.spans and spans is not None:

            # Initialise document spans
            doc_spans = list()

            # Loop over all different span types
            for label, value in spans.items():
                for start, end in value['spans']:
                    # Add document span
                    doc_spans.append(result.char_span(start, end,
                        label          = label,
                        alignment_mode = 'expand',
                    ))

            # Set document entities
            document.set_ents(
                spacy.util.filter_spans(
                    tuple(filter(None, spans)) +
                    document.ents
                )
            )

        # Return result
        return result

    ########################################################################
    #                          Extraction methods                          #
    ########################################################################

    def pdf(self, data):
        """Extract text from PDF data.

            Parameters
            ----------
            data : bytes
                Bytes of PDF.

            Returns
            -------
            text : string
                Text extracted from PDF.
            """
        # Check if pdfact is set
        if self.pdfact is None:
            raise ValueError(
                "PDF extraction requires a specified path to pdfact"
            )

        ####################################################################
        #                        Read text from PDF                        #
        ####################################################################

        # Temporarily write bytes to file
        with tempfile.NamedTemporaryFile() as outfile:
            # Write PDF to temporary file
            outfile.write(data)

            # Get filename of temporary file
            infile = os.path.realpath(outfile.name)

            # Invoke PDFact on input file
            process = subprocess.Popen(
                ["java", "-jar", self.pdfact, infile],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Get output
            out, err = process.communicate()

            # Raise error in case of error
            if err:
                warnings.warn(
                    "pdfact Had an error:\n" +
                    err.decode('utf-8') +
                    "\nTrying pdftotext conversion..."
                )

                # Invoke PDFact on input file
                process = subprocess.Popen(
                    ["pdftotext", infile, '-'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Get output
                out, err = process.communicate()

                if err:
                    raise ValueError(err.decode('utf-8'))

            # Decode output
            result = out.decode('utf-8')

        # Ensure sentences are completed (with period)
        result = '\n\n'.join([
            f"{paragraph}."
            if paragraph and paragraph[-1] not in '.,?!:;\'"()[]{}'
            else paragraph
            for paragraph in self.regex_paragraphs.split(result)
        ])

        # Return result
        return result


    def html(self, html):
        """Extract text from HTML input.

            Parameters
            ----------
            html : string | bytes
                HTML data to convert.

            Returns
            -------
            text : string
                Text extracted from HTML file.

            spans : dict
                Dictionary of character spans retrieved from HTML.
            """
        # Ensure html is given as text
        if isinstance(html, bytes):
            html = html.decode('utf-8')

        # Use HTML preprocessor to extract text from HTML
        return self.preprocessor.html(html)


    def ascii(self, text):
        """Extract text from ASCII text.
            Currently does nothing.

            Parameters
            ----------
            text : string | bytes
                ASCII text from which to extract text.

            Returns
            -------
            text : string
                Text extracted from ASCII text.
            """
        # Ensure text is given as string
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        # Return text
        return text


    def zip(self, infile):
        """Extract text from ZIP infile.

            Parameters
            ----------
            infile : string
                Path to ZIP input file.

            Returns
            -------
            text : string
                Text extracted from ZIP file.
            """
        raise ValueError("Not yet implemented properly")
        # Initialise result
        result = list()

        ####################################################################
        #                        Read text from ZIP                        #
        ####################################################################

        # Read text from ZIP file
        zf = zipfile.ZipFile(infile)

        # Loop over all PDFs in ZIP archive
        for pdf in [name for name in zf.namelist() if name.endswith('pdf')]:
            # Extract PDF file
            zf.extract(pdf)
            # Extract text from pdf file
            result.append(self.pdf(pdf))

            # Remove extracted PDF file
            if os.path.exists(pdf):
                os.remove(pdf)
            else:
                warnings.warn(f"Could not delete '{pdf}'")

        # Return text
        return "\n\n".join(result)
