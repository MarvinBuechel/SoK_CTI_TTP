from typing import List, Optional, Tuple
from spacy.tokens import Doc, Span, Token

class DependencyError(Exception):
    """DependencyError gives an overview of missing dependencies."""

    def __init__(
            self,
            name: str,
            dependencies: List[Tuple[str, str, bool]],
            *args, **kwargs,
        ):
        """DependencyError listing the missing dependencies."""
        # Initialise Exception
        super().__init__(args, **kwargs)

        self.name = name
        self.dependencies = dependencies
    
    def __str__(self):
        """Get string representation of exception."""
        # Initialise result
        result = f"\n\nSpaCy pipe '{self.name}' misses required extensions.\n"

        registered = list(filter(lambda x:     x[2], self.dependencies))
        missing    = list(filter(lambda x: not x[2], self.dependencies))

        # List registered extensions
        if registered:
            result += "\nThe following required exstensions were successfully registered:\n"
            for extension, pipe, _ in registered:
                result += f'\t✓ {extension} ({pipe})\n'

        # List missing extensions
        result += "\nThe following required exstensions are missing:\n"
        for extension, pipe, _ in missing:
            result += f'\t✘ {extension} ({pipe})\n'

        # Return result
        return result


def check_dependencies(
        nlp,
        name: str,
        doc  : Optional[List[Tuple[str, str]]] = None,
        span : Optional[List[Tuple[str, str]]] = None,
        token: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
    """Check whether pipeline contains required dependencies.
    
        Parameters
        ----------
        nlp : spacy.Language
            Pipeline on which to check dependencies.

        name : str
            Name of current pipe for which to check dependencies.

        doc : Optional[List[Tuple[str, str]]]
            List of required Doc extensions to be set.
        
        span : Optional[List[Tuple[str, str]]]
            List of required Span extensions to be set.
        
        token : Optional[List[Tuple[str, str]]]
            List of required Token extensions to be set.
        """
    # Initialise dependencies
    dependencies = list()

    # Get nlp pipes
    pipes = set(dict(nlp.pipeline))

    # Check Doc dependencies
    if doc is not None:
        for dependency, pipe in doc:
            dependencies.append((
                dependency,
                pipe,
                Doc.has_extension(dependency) and pipe in pipes,
            ))

    # Check Span dependencies
    if span is not None:
        for dependency, pipe in span:
            dependencies.append((
                dependency,
                pipe,
                Span.has_extension(dependency) and pipe in pipes,
            ))

    # Check Token dependencies
    if token is not None:
        for dependency, pipe in token:
            dependencies.append((
                dependency,
                pipe,
                Token.has_extension(dependency) and pipe in pipes,
            ))

    # Raise error if
    if any(not registered for dependency, pipe, registered in dependencies):
        raise DependencyError(name, dependencies)

    # Return true
    return True