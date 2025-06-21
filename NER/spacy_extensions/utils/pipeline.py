from spacy.language import Language
from typing import Iterable

# Base pipes for CTI pipeline
base_pipes = {
    'transformer',
    'tok2vec',
    'tagger',
    'parser',
    'attribute_ruler',
    'lemmatizer',
    'matcher_ioc',
    'matcher_mitre_attack',
    'merge_entities',
}

def nlp2preprocessed(
        nlp: Language,
        disable_pipes: Iterable[str] = base_pipes,
    ) -> Language:
    """Convert NLP pipeline to a version dealing with preprocessed docs.
        This is the inverse functionality of :py:meth:`preprocessed2nlp`.

        Note
        ----
        When applying this function to a pipeline, please ensure that the docs
        that you feed as input have been preprocessed using a base pipe.
        If not, the pipeline will throw errors when trying to process raw text
        or non-preprocessed documents.

        Parameters
        ----------
        nlp : spacy.Language
            Pipeline to convert to a pipeline for handling preprocessed docs.

        disable_pipes : Iterable[str], default = base_pipes
            Pipes to disable.

        Returns
        -------
        nlp : spacy.Language
            Pipeline where base functionality has been disabled to deal with
            preprocessed docs faster.
        """
    # Loop over all components in pipeline
    for component in nlp.component_names:
        if component in disable_pipes:
            nlp.disable_pipe(component)

    # Return pipeline
    return nlp


def preprocessed2nlp(
        nlp: Language,
        enable_pipes: Iterable[str] = base_pipes,
    ) -> Language:
    """Convert NLP pipeline that deals with preprocessed docs to a version
        dealing with regular text or non-preprocessed docs. This is the inverse
        functionality of :py:meth:`nlp2preprocessed`.

        Parameters
        ----------
        nlp : spacy.Language
            Pipeline to convert to a pipeline for handling preprocessed docs.

        enable_pipes : Iterable[str], default = base_pipes
            Pipes to enable.

        Returns
        -------
        nlp : spacy.Language
            Pipeline where base functionality has been re-enabled to deal with
            normal docs or text.
        """
    # Loop over all pipes in pipeline
    for component in nlp.component_names:
        if component in enable_pipes:
            nlp.enable_pipe(component)

    # Return pipeline
    return nlp


def nlp2base(
        nlp: Language,
        base_pipes: Iterable[str] = base_pipes,
    ) -> Language:
    """Convert NLP pipeline to its base form for preprocessing. This is the
        inverse of :py:meth:`base2nlp`.

        Parameters
        ----------
        nlp : spacy.Language
            Pipeline to convert to a pipeline for handling preprocessed docs.

        base_pipes : Iterable[str], default = base_pipes
            Pipes to keep enabled.

        Returns
        -------
        nlp : spacy.Language
            Pipeline where only base components are enabled.
        """
    # Loop over all pipes in pipeline
    for component in nlp.component_names:
        if component not in base_pipes:
            nlp.disable_pipe(component)

    # Return pipeline
    return nlp


def base2nlp(
        nlp: Language,
        base_pipes: Iterable[str] = base_pipes,
    ) -> Language:
    """Convert base NLP pipeline to its full form again. This is the
        inverse of :py:meth:`nlp2base`.

        Note
        ----
        If pipes have been disabled before :py:meth:`nlp2base` was called, the
        pipes will be re-enabled by this method.

        Parameters
        ----------
        nlp : spacy.Language
            Pipeline to convert to a pipeline for handling preprocessed docs.

        base_pipes : Ignored
            Ignored

        Returns
        -------
        nlp : spacy.Language
            Pipeline where only base components are enabled.
        """
    # Loop over all pipes in pipeline
    for component in nlp.component_names:
        nlp.enable_pipe(component)

    # Return pipeline
    return nlp
