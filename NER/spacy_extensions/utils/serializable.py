# Imports
from __future__ import annotations
from pathlib    import Path
from typing     import Iterable, Optional, Union

class SpacySerializable:
    """Object provides default serializable methods for SpaCy pipes.
        Adds to_disk() and from_disk() methods to any object that implements
        to_bytes() and from_bytes() methods.
    """

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Serialize pipe to bytes.

            Parameters
            ----------
            exclude : Optional[Iterable[str]]
                String names of serialization fields to exclude.

            Returns
            -------
            result : bytes
                Serialized object.
            """
        raise NotImplementedError(
            "pipe.to_bytes() should be implemented by subclass."
        )


    def from_bytes(
            self,
            data: bytes,
            exclude: Optional[Iterable[str]] = None,
        ) -> SpacySerializable:
        """Load pipe from bytes in place.

            Parameters
            ----------
            data : bytes
                Data from which to load pipe.

            exclude : Optional[Iterable[str]]
                String names of serialization fields to exclude.

            Returns
            -------
            result : self
                Return pipe modified in place.
            """
        raise NotImplementedError(
            "pipe.from_bytes() should be implemented by subclass."
        )


    def to_disk(
            self,
            path   : Union[str, Path],
            exclude: Optional[Iterable[str]] = None,
        ) -> None:
        """Serialize the pipe to disk.

            Parameters
            ----------
            path : Union[str, Path]
                A path to a directory, which will be created if it doesn’t
                exist. Paths may be either strings or Path-like objects.

            exclude : Iterable[str], optional
                String names of serialization fields to exclude.
            """
        # Transform path to Path
        if isinstance(path, str): path = Path(path)

        # Ensure file path exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Open file
        with open(path, 'wb') as outfile:
            # Write bytes to outfile
            outfile.write(self.to_bytes(exclude=exclude))


    def from_disk(
            self,
            path: Union[str, Path],
            exclude: Optional[Iterable[str]] = None,
        ) -> SpacySerializable:
        """Load the pipe from disk. Modifies the object in place.

            Parameters
            ----------
            path : Union[str, Path]
                A path to a file. Paths may be either strings or Path-like
                objects.

            exclude : Iterable[str], optional
                String names of serialization fields to exclude.

            Returns
            -------
            result : self
                Return pipe modified in place.
            """
        # Open file
        with open(path, 'rb') as infile:
            # Load SynonymFinder from bytes of infile and return
            return self.from_bytes(infile.read())
