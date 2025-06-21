# Imports
from spacy.tokens import Doc
from typing       import Any, Iterable

class SpacyCallable:
    """Object provides default callable methods for SpaCy pipes.
        Adds __call__() and pipe() methods to any object.
    """

    def __call__(self, document: Doc) -> Doc:
        """Default pipe callable, does not modify the document in any way.

            Note
            ----
            This method can be overwritten in subclasses. By doing so, the
            pipe automatically inherrits the ``pipe()`` method. See ``pipe()``
            for functionality.

            Parameters
            ----------
            document : Doc
                Document for which to run the pipe.

            Returns
            -------
            document : Doc
                Unmodified document.
            """
        return document


    def pipe(
            self,
            documents: Iterable[Doc],
             *args   : Any,
            **kwargs : Any,
        ) -> Iterable[Doc]:
        """Runs multiple documents through the __call__ method. By default it
            provides an iterator that yields the ``__call__(document)`` output.

            Note
            ----
            This method can be overwritten in subclasses. By doing so, one can
            store intermediate values in the pipe object or control e.g.,
            batching of documents. If you want to change the actual behaviour of
            the pipe, please modify the ``__call__()`` method instead.

            Parameters
            ----------
            documents : Iterable[Doc]
                Documents to run through the pipe.

            *args : Any
                Ignored

            **kwargs : Any
                Ignored

            Yields
            ------
            document : Doc
                Document run through __call__(document).
            """
        for document in documents:
            yield self(document)
