from enum import Enum
from spacy.tokens import Span, Token
from typing import Iterable, Iterator, Optional, Tuple
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import spacy
import warnings

class Dependency(Enum):
    SUBJECT = 'subject'
    OBJECT  = 'object'
    VERB    = 'verb'
    UNKNOWN = None


class SentenceTree(object):

    def __init__(self, sentence: Span):
        """Create a sentence tree from a SpaCy-parsed sentence.

            Parameters
            ----------
            sentence : Span
                Sentence for which to generate the tree structure.
            """
        # Ensure sentence is correct
        assert isinstance(sentence, spacy.tokens.span.Span), "sentence should be of type spacy.tokens.span.Span"

        # Store sentence
        self.sentence = sentence

        # Create a directed graph
        self.tree = self.to_tree(self.sentence.root)

        # Check if tree is actually a tree
        assert nx.algorithms.tree.recognition.is_tree(self.tree), "Sentence is not a tree"

    ########################################################################
    #                     Extract sentence properties                      #
    ########################################################################

    def entities(
            self,
            filter: Optional[Iterable[str]] = None,
        ) -> Iterator[Span]:
        """Iterable over entities from sentence.

            Parameters
            ----------
            filter : Optional[Iterable[str]]
                If given, only return entity labels present in filter.

            Yields
            ------
            entity : Span
                Entity span in sentence.
            """
        # Iterate over entities
        for entity in self.sentence.ents:
            # Apply filter if required
            if filter is None or self.sentence.doc.vocab[entity.label].text in filter:
                # Yield entity
                yield entity


    def entity_roots(
            self,
            filter: Optional[Iterable[str]] = None,
        ) -> Iterator[Tuple[Span, Token]]:
        """Iterate over all entity roots in sentence.

            Parameters
            ----------
            filter : Optional[Iterable[str]]
                If given, only apply function to  entity labels present in
                filter. Also see sentence_tree.SentenceTree.entities().

            Yields
            ------
            entity : Span
                Entity span in sentence.

            root : Token
                Root token of entity.
            """
        # Iterate over all entities
        for entity in self.entities(filter=filter):
            # Get tokens from entity
            tokens = list(entity)

            # If entity consists of single token, yield token
            if len(tokens) == 1:
                yield entity, tokens[0]

            # Otherwise find root of tokens
            else:
                # The root is the lowest common ancestor of all tokens in entity
                yield entity, self.lowest_common_ancestor(tokens)


    def entity_dependency(
            self,
            root: Token,
            default: Dependency = Dependency.UNKNOWN,
        ) -> Dependency:
        """Get the dependency of the entity, given its entity root.

            Parameters
            ----------
            root : spacy.tokens.token.Token
                Root token of entity.
                See sentence_tree.SentenceTree.entity_roots() on how to retrieve
                an entity root.

            default : Dependency, default=Dependency.UNKNOWN
                Default object to return if no dependency is found.

            Returns
            -------
            dependency : Dependency
                Dependency of token within sentence.
                See Dependency class on the available dependencies.
            """
        # Get node from root
        parents = list(self.tree.in_edges(hash(root), data=True))

        # Traverse the tree upwards while we still have parents
        while len(parents) > 0:

            # Ensure we have a maximum of 1 parent
            assert len(parents) == 1, "Parents node {} ({}) != 1".format(
                hash(root),
                self.tree.nodes[hash(root)].get('token'),
            )

            # Get single parent
            parent, root_, data = parents[0]

            # Ensure roots are equal
            assert root_ == hash(root), "Found different root"

            if 'subj' in data.get('dep_'):
                return Dependency.SUBJECT
            elif 'obj' in data.get('dep_'):
                return Dependency.OBJECT

            # Get node from root
            root    = self.tree.nodes[parent].get('token')
            parents = list(self.tree.in_edges(hash(root), data=True))

        # Return default
        return default


    def sentence_tuples(self, filter=None):
        """Iterate over the (subject, verb, object) tuples of the sentence.

            Parameters
            ----------
            filter : iterable of string, optional
                If given, only apply function to  entity labels present in
                filter. Also see sentence_tree.SentenceTree.entities().

            Yields
            ------
            subject : spacy.tokens.span.Span
                Subject in tuple.

            verb : list of spacy.tokens.span.Span
                Verb(s) connecting subject and object.

            object : spacy.tokens.span.Span
                Object in tuple.
            """
        # Get list of subjects and objects
        subjects = set()
        objects  = set()

        # Loop over all entities
        for entity, root in self.entity_roots(filter=filter):
            # Get dependency
            dependency = self.entity_dependency(root)

            # Add to subject or object
            if dependency == Dependency.SUBJECT:
                subjects.add((entity, root))
            elif dependency == Dependency.OBJECT:
                objects.add((entity, root))
            else:
                warnings.warn(f"Unknown entity found: {entity} ({dependency})")

        # Loop over all subject - object combinations
        for (subject, sroot), (object, oroot) in itertools.product(subjects, objects):
            # Get root connecting subject to object
            root = self.lowest_common_ancestor((sroot, oroot))

            # Find all verbs in the path
            spath = list()
            for ancestor in sroot.ancestors:
                spath.append(ancestor)
                if ancestor == root: break

            opath = list()
            for ancestor in oroot.ancestors:
                opath.append(ancestor)
                if ancestor == root: break

            verbs = [
                token
                for token in spath[:-1] + opath[::-1]
                if token.pos_ == 'VERB'
            ]

            # Yield subject, verb(s), object
            yield subject, verbs, object

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_tree(self, root, tree=None):
        """Create a networkx tree from a given sentence root.

            Parameters
            ----------
            root : spacy.tokens.token.Token
                The root token from which to generate the tree.

            tree : nx.DiGraph, (optional)
                Existing tree for which to root children. Note that this is used
                to recurrently build up the tree and should not be used as an
                input parameter.

            Returns
            -------
            tree : nx.DiGraph
                Tree as build up from the sentence root.
            """
        # Create tree if it does not yet exist
        if tree is None:
            # Create tree
            tree = nx.DiGraph()
            # Add root node
            tree.add_node(hash(root), token=root)

        # Loop over children
        for child in root.children:
            # Add child node
            tree.add_node(hash(child), token=child)
            # Add child edge
            tree.add_edge(hash(root), hash(child), dep_=child.dep_)

            # Recursively add children
            tree = self.to_tree(child, tree)

        # Return tree
        return tree

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def as_tree(self, root=None):
        """Create a networkx tree from a given sentence root.

            Parameters
            ----------
            root : spacy.tokens.token.Token (optional)
                The root token from which to generate the tree.

            tree : nx.DiGraph, (optional)
                Existing tree for which to root children. Note that this is used
                to recurrently build up the tree and should not be used as an
                input parameter.

            Returns
            -------
            tree : nx.DiGraph
                Tree as build up from the sentence root.
            """
        # Set root if not given
        if root is None:
            root = self.sentence.root

        # Initialise tree
        tree = {
            'token': root.text,
            'dep'  : root.dep_,
            'ent'  : root.ent_type_,
            'full' : '',
        }

        # Add children
        children = list()
        # Loop over children
        for child in root.children:
            if child.text.strip():
                children.append(self.as_tree(child))

        # Add children if any
        if children:
            tree['children'] = children

        # Set full text
        for token in sorted(root.subtree, key=lambda x: x.i):
            tree['full'] += token.text
            if token.whitespace_:
                tree['full'] += ' '

        # Return tree
        return tree


    def show(self, dep=False, ner=False, pos=False, iocs=None):
        """Show the sentence in a tree structure.

            Parameters
            ----------
            dep : boolean, default=False
                If True, show dependencies between tokens.

            ner : boolean, default=False
                If True, show named entity recognition of tokens.

            pos : boolean, default=False
                If True, show part of speach for each token.

            iocs : iterable, optional
                If given, only highlight the named entities in the iocs.
            """
        ####################################################################
        #                          Tree structure                          #
        ####################################################################

        # Create position tree structure
        position = nx.nx_agraph.graphviz_layout(self.tree, prog="dot")

        ####################################################################
        #                            Draw tree                             #
        ####################################################################

        # Set node color
        node_color = '#ffffff'
        if ner and iocs:
            node_color = [
                '#888888' if data.get('token').ent_type_ in iocs else '#ffffff'
                for node, data in self.tree.nodes(data=True)
            ]
        elif ner:
            node_color = [
                '#888888' if data.get('token').ent_type_ else '#ffffff'
                for node, data in self.tree.nodes(data=True)
            ]

        # Draw tree
        nx.draw_networkx(
            self.tree,
            pos        = position,
            labels     = {
                node: data.get('token')
                for node, data in self.tree.nodes(data=True)
            },
            node_color = node_color,
            node_shape = 's',
            node_size  = 1000,
        )

        # Only show dependencies if required
        if dep:
            nx.draw_networkx_edge_labels(
                self.tree,
                pos         = position,
                edge_labels = {
                    (u, v): data.get('dep_')
                    for u, v, data in self.tree.edges(data=True)
                },
                font_size   = 8,
            )

        # Only show part of speach if required
        if pos:
            nx.draw_networkx_labels(
                self.tree,
                pos       = {k: (v[0], v[1]-15) for k, v in position.items()},
                labels    = {
                    node: data.get('token').pos_
                    for node, data in self.tree.nodes(data=True)
                },
                font_size = 8,
            )

        ####################################################################
        #                              Title                               #
        ####################################################################

        # Split longer titles
        title = list()
        sub_title = ""
        start = 0
        end   = 0
        while end < len(self.sentence):
            while end < len(self.sentence) and len(sub_title) < 100:
                sub_title = self.sentence[start:end+1].text
                end += 1
            title.append(sub_title)
            sub_title = ""
            start     = end-1

        # Set title
        plt.title('\n'.join(title))

        ####################################################################
        #                            Show tree                             #
        ####################################################################

        plt.show()


    ########################################################################
    #                         Auxiliary functions                          #
    ########################################################################

    def lowest_common_ancestor(self, tokens):
        """Find the lowest common ancestor of all given tokens.

            Parameters
            ----------
            tokens : iterable of spacy.tokens.token.Token
                Tokens for which to find the lowest common ancestor.

            Returns
            -------
            root : spacy.tokens.token.Token
                Lowest common ancestor of all given tokens.
            """
        # Get nodes from tokens
        nodes = list(map(hash, tokens))

        # Loop over all nodes until we find a root
        while len(nodes) >= 2:

            # Get root of next two nodes
            root = nx.algorithms.lowest_common_ancestor(
                G     = self.tree,
                node1 = nodes.pop(),
                node2 = nodes.pop(),
            )

            # Ensure root is not None
            assert root is not None, "Could not find lowest_common_ancestor."

        # Insert root node
        nodes.append(root)

        # Return root node
        return self.tree.nodes[nodes[0]]['token']


################################################################################
#                            SpaCy AnalyzerSentence                            #
################################################################################

@spacy.Language.factory('analyzer_sentence')
class AnalyzerSentence(object):

    def __init__(self, nlp, name):
        """AnalyzerSentence for analyzing text on sentence level.
            New sentences are available as `document._.sentences`.
            """
        # Set SpaCy pipeline objects
        self.nlp  = nlp
        self.name = name

    ########################################################################
    #                                 TODO                                 #
    ########################################################################

    def __call__(self, document):
        """The AnalyzerSentence creates additional attributes for each sentence
            in the document. New sentences are available as
            `document._.sentences`.

            Parameters
            ----------
            document : spacy.Doc
                Document for which to add sentence attributes.

            Returns
            -------
            document : spacy.Doc
                Document where `document._.sentences` are equivalent to
                document.sents with the following additional attributes:
                - `._.tree`: Dictionary representing tree as sentence.
            """
        # Check if sentence extension already exists
        if not document.has_extension('sentences'):
            # Set sentence extension
            document.set_extension(
                name   = 'sentences',
                getter = self.sentences,
            )

        # Return document
        return document


    def sentences(self, document):
        """Create new sentences"""
        # Loop over original sentences
        for sentence in document.sents:
            # Parse sentence using sentence tree
            sentence_tree = SentenceTree(sentence)

            # Add properties
            sentence.set_extension(
                name    = 'tree',
                default = sentence_tree.as_tree(),
                force   = True,
            )

            # Add properties
            sentence.set_extension(
                name    = 'sentence',
                default = sentence_tree,
                force   = True,
            )

            # Yield sentence
            yield sentence
