from itertools import chain, product
from typing import Iterator, List, Optional, Tuple
from spacy                  import Language
from spacy.tokens           import Doc, Span, Token
from spacy_extensions.utils import SpacyCallable
import matplotlib.pyplot as plt
import networkx as nx

################################################################################
#                              Auxiliary methods                               #
################################################################################

def span2tree(span: Span) -> nx.DiGraph:
    """Transform a span to a tree represented by a nx.DiGraph.
    
        Parameters
        ----------
        span : Span
            Span to transform to a tree.
            
        Returns
        -------
        tree : nx.DiGraph
            Tree representation of span.
        """
    # Return tree based on root token
    return token2tree(span.root)

def token2tree(token: Token, tree: Optional[nx.DiGraph] = None) -> nx.DiGraph:
    """Create a networkx tree from a given root token.

        Parameters
        ----------
        token : spacy.tokens.token.Token
            The root token from which to generate the tree.

        tree : nx.DiGraph, (optional)
            Existing tree for which to generate children. Note that this is used
            to recursively build up the tree and should not be used as an input
            parameter.

        Returns
        -------
        tree : nx.DiGraph
            Tree as build up from the token root.
        """
    # Create tree if it does not yet exist
    if tree is None:
        # Create tree
        tree = nx.DiGraph()
        # Add root node
        tree.add_node(hash(token), token=token)

    # Loop over children
    for child in token.children:
        # Add child node
        tree.add_node(hash(child), token=child)
        # Add child edge
        tree.add_edge(hash(token), hash(child), dep_=child.dep_)

        # Recursively add children
        tree = token2tree(child, tree)

    # Return tree
    return tree

def show_tree(span: Span) -> None:
    """Show the given span in a tree format.
    
        Parameters
        ----------
        span : Span
            Span to show in tree format.
    """
    # Get tree from span
    tree = span2tree(span)

    # Create position tree structure
    position = nx.nx_agraph.graphviz_layout(tree, prog="dot")

    ####################################################################
    #                            Draw tree                             #
    ####################################################################

    # Set node color
    node_color = '#ffffff'
    ner_color  = '#888888'
    node_color = [
        ner_color if data.get('token').ent_type_ else node_color
        for _, data in tree.nodes(data=True)
    ]

    # Draw tree
    nx.draw_networkx(
        tree,
        pos        = position,
        node_color = node_color,
        node_shape = 's',
        node_size  = 1000,
        labels     = {
            node: data.get('token')
            for node, data in tree.nodes(data=True)
        },
    )

    # Draw dependencies
    nx.draw_networkx_edge_labels(
        tree,
        pos         = position,
        font_size   = 8,
        edge_labels = {
            (u, v): data.get('dep_')
            for u, v, data in tree.edges(data=True)
        },
    )

    # Draw part of speech
    nx.draw_networkx_labels(
        tree,
        pos       = {k: (v[0], v[1]-15) for k, v in position.items()},
        font_size = 8,
        labels    = {
            node: data.get('token').pos_
            for node, data in tree.nodes(data=True)
        },
    )

    ####################################################################
    #                            Show tree                             #
    ####################################################################

    # Set title
    plt.title(f'"{span.text.strip()}"')
    # Hide axes
    plt.axis('off')
    # Plot tree
    plt.show()


def tree_traverse(
        tree: nx.DiGraph,
        source: object,
        target: object,
    ) -> Iterator[object]:
    """Traverse a tree from the given source to target node.
    
        Parameters
        ----------
        tree : nx.DiGraph
            Tree to traverse.
            
        source : object
            Node in tree to start traversal (child node).
            
        target : object
            Node in tree to stop traversal (ancestor).
            
        Yields
        ------
        node : object
            Node found while traversing.
        """
    # Yield source
    yield source
    # Stop if source == target
    if source == target: return

    # Get all parents
    parents = list(tree.predecessors(source))

    # Assert there is only a single parent
    if len(parents) > 1:
        raise ValueError(
            f"Multiple parents found for node '{source}' in tree '{tree}.' "
            "Nodes in tree should only have one parent."
        )

    # Case when target was not found
    if len(parents) == 0:
        raise ValueError(
            f"Target '{target}' not found while traversing tree '{tree}'."
        )

    # Yield parents
    yield from tree_traverse(tree, source=parents[0], target=target)

################################################################################
#                                   SVO pipe                                   #
################################################################################

@Language.factory('svo')
class SVO(SpacyCallable):
    """The SVO pipe provides Span and Doc extensions for retrieving the
        ``(subject, verb, object)`` tuples of document sentences and spans.

        Extensions
        ----------
        svos : Doc
            Iterates over all (subject, verb, object) tuples for each sentence
            in document. See :py:meth:`sentence_svo`.

        svo : Span
            Returns list of all (subject, verb, object) tuples for given span.
            See :py:meth:`svo`.

        tree : Span
            Represents span dependencies as nx.DiGraph tree. See
            :py:meth:`span2tree`.

        subtree_text : Token
            Returns the text of entire token subtree. See
            :py:meth:`subtree_text`.
        """
    
    def __init__(self, nlp, name: str):
        """SVO constructor.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which SVO is used.

            name : string
                Name of pipeline.
            """
        # Initialize pipeline
        self.nlp  = nlp
        self.name = name

        # Register doc extensions
        Span.set_extension(
            name   = "tree",
            getter = lambda span: span2tree(span),
        )

        Span.set_extension(
            name = "show_tree",
            getter = lambda span: show_tree(span),
        )

        Span.set_extension(
            name   = "svo",
            getter = self.svo,
        )

        Doc.set_extension(
            name   = "svos",
            getter = self.sentence_svo,
        )

        Token.set_extension(
            name = 'subtree_text',
            getter = self.subtree_text
        )
    

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def subtree_text(self, token: Token) -> str:
        """Get text of token subtree.
        
            Parameters
            ----------
            token : Token
                Token for which to get the full subtree text.
            
            Returns
            -------
            text : str
                Text of full token subtree.
            """
        # Get subtree as list
        subtree = list(token.subtree)
        for index, token in enumerate(subtree):
            if index == len(subtree) -1:
                subtree[index] = token.text
            else:
                subtree[index] = f"{token.text}{token.whitespace_}"

        # Return joined subtree
        return ''.join(subtree)
            

    def svo(self, span: Span) -> List[Tuple[Token, Token, Token]]:
        """Returns subject-verb-object tuples for a given span.
        
            Parameters
            ----------
            span : Span
                Span for which to retrieve subject-verb-object tuples.
                
            Returns
            -------
            svo : List[Tuple[Token, Token, Token]]
                List of subject-verb-object tuples found in span.
            """
        # Initialise result
        result = list()

        # Transform the span into a tree format
        tree = span._.tree

        # Initialise subject, verb and object
        subjects = set()
        objects  = set()

        # Search subject and object
        for token in span:
            if 'subj' in token.dep_:
                subjects.add(token)
            elif 'obj' in token.dep_:
                objects.add(token)

        # For each subject, object combination, find corresponding verb
        for subject, object in product(subjects, objects):
            # Find lowest common ancestor of both nodes
            ancestor = nx.lowest_common_ancestor(
                G = tree,
                node1 = hash(subject),
                node2 = hash(object),
            )

            # Get iterator over related nodes
            nodes = chain(
                tree_traverse(tree, hash(subject), ancestor),
                tree_traverse(tree, hash(object ), ancestor),
            )
            # Map nodes to tokens
            tokens = map(lambda node: tree.nodes[node]['token'], nodes)
            # Filter tokens of type VERB
            verbs = set(filter(lambda token: token.pos_ == 'VERB', tokens))

            # Add found tuples to result
            for verb in verbs:
                result.append((subject, verb, object))

        # Return result
        return result
        

    def sentence_svo(self, doc: Doc) -> Iterator[List[Tuple[Span, Span, Span]]]:
        """Yields list of subject-verb-object tuples for each sentence in
            document.
            
            Parameters
            ----------
            doc : Doc
                Document for which to yield list of subject-verb-object tuples
                for each sentence.

            Yields
            ------
            svos : List[Tuple[Span, Span, Span]]
                Subject-verb-object tuples found for each sentence in doc.
                See :py:meth:`svo` for more details.
            """
        for sentence in doc.sents:
            yield self.svo(sentence)
