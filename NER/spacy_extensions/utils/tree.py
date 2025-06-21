"""Tree utilities for iterating over tokens in a sentence.

    Example
    -------
    ```python

    # Register token extensions
    Token.set_extension('dfs', getter=lambda x: dfs(x))
    Token.set_extension('bfs', getter=lambda x: bfs(x))

    # Register span extensions
    Span.set_extension('dfs', getter=lambda x: dfs(x.root))
    Span.set_extension('bfs', getter=lambda x: bfs(x.root))
    ```
"""
# Imports
from collections import deque
from itertools import product
from spacy.tokens import Span, Token
from typing import Any, Iterable, Iterator, List, Optional, Set
import networkx as nx


def dfs(root: Token) -> Iterator[Token]:
    """Perform a depth first search over a parsed span root.
    
        Parameters
        ----------
        root : Token
            Root token from which to start depth first search.
            
        Yields
        ------
        token : Token
            Next token in depth first search.
        """
    # Yield token itself
    yield root
    # Loop over all children
    for child in root.children:
        # Recursively perform depth first search
        yield from dfs(child)


def bfs(root: Token) -> Iterator[Token]:
    """Perform a breadth first search over a parsed span root.
    
        Parameters
        ----------
        root : Token
            Root token from which to start breadth first search.
            
        Yields
        ------
        token : Token
            Next token in breadth first search.
        """
    # Create queue
    q = deque([root])

    # Loop while queue is not empty
    while len(q):
        # Get next token
        token = q.popleft()
        # Yield token
        yield token

        # Add token children to queue
        for child in token.children:
            q.append(child)


def root2tree(root: Token, tree: Optional[nx.DiGraph]=None):
    """Create a networkx tree from a given token root.
        Assumes that dependencies have already been parsed.
    
        Parameters
        ----------
        root : Token
            Root from which to generate tree.
            
        tree : Optional[nx.DiGraph]
            Used for recursive calling, should not be called directly.

        Returns
        -------
        tree : nx.DiGraph
            Tree built from root. 
        """
    # Create tree if none is given
    if tree is None:
        tree = nx.DiGraph()
    
    # Loop over all root children
    for child in root.children:
        # Add root -> child dependency
        tree.add_edge(root, child, dep_=child.dep_)
        # Recursively add children to tree
        tree = root2tree(child, tree)
    
    # Return tree
    return tree


def subtree_view(
        tree: nx.DiGraph,
        nodes: Iterable[Any],
    ) -> nx.DiGraph:
    """Create a minimal subtree that includes all nodes.
    
        Parameters
        ----------
        tree : nx.DiGraph
            Graph for which to create minimal subtree.

        root : Optional[Any]
            Root node of tree

        nodes : Iterable[Any]
            Nodes in tree for which to create minimal subtree.

        Returns
        -------
        subtree : View of nx.DiGraph
            Minimal subtree. Note that this is a SubGraph View, hence attributes
            changd within subtree will be affecting the original tree. To make a
            copy of the tree call subtree.copy(), this copy can be modified
            without affecting the original tree.
        """
    # In case the tree is empty, return an empty view
    if len(tree) == 0:
        return tree

    # Get set of nodes to include in tree
    included = set()
    required = set(nodes)
    current = set()

    # Look for required nodes, starting from bottom of tree
    for layer in reversed(tree_hierarchy(tree)):
        # Find required nodes in layer
        found_nodes = required & layer
        # Remove found nodes from required
        required -= found_nodes

        # Get list of current nodes
        current |= found_nodes
        included |= current

        # Stop if we found everything
        if not required and len(current) <= 1:
            included |= current
            break

        # Otherwise, get parents of current nodes
        else:
            current = map(tree.predecessors, current)
            current = set(map(next, current))
            current = set(current)

    assert set(nodes).issubset(included), "Not all nodes included"
    # Create and return subtree
    return tree.subgraph(included)
       

def tree_hierarchy(tree: nx.DiGraph) -> List[Set[Any]]:
    # Get root of tree
    root = [n for n,d in tree.in_degree() if d==0]
    # Find layers
    return list(map(set, nx.bfs_layers(tree, root[0])))


def subtrees(
        sentence: Span,
        includes: List[Iterable[Token]],
        allow_duplicate: bool = False,
    ) -> Iterator[Span]:
    """Identify all subtrees in a sentence Span that include at least one of
        each token in the includes list.

        Note
        ----
        Looks like a variation on:
        https://en.wikipedia.org/wiki/Steiner_tree_problem

        Parameters
        ----------
        sentence : Span
            Sentence in which to search for tokens.

        includes : List[Iterable[Token]]
            List of tokens that should be included. Should include at least one
            value from each outer list.

        allow_duplicate : bool, default=False
            If True, allow duplicate use of tokens in include if tokens appear
            in multiple lists.

        Yields
        ------
        subtree : Span
            Possible subtree spanning at least one of each Token in includes.
        """
    # Assert that all tokens are present in sentence
    for lists in includes:
        for token in lists:
            if token not in sentence:
                raise ValueError(
                    f"Token '{token}' not in sentence \"{sentence}\"."
                )
    
    # Convert includes to token indices
    includes = [[token.i for token in lists] for lists in includes]

    # Loop over all products of includes
    for combination in product(*includes):
        # Skip duplicates if required
        if not allow_duplicate and len(combination) != len(set(combination)):
            continue

        # Yield subtree
        yield Span(sentence.doc, min(combination), max(combination)+1)


def min_subtree(
        sentence: Span,
        includes: List[Iterable[Token]],
        allow_duplicate: bool = False,
    ) -> Optional[Span]:
    """Identify the minimum subtree in a sentence Span that include at least
        one of each token in the includes list.

        Parameters
        ----------
        sentence : Span
            Sentence in which to search for tokens.

        includes : List[Iterable[Token]]
            List of tokens that should be included. Should include at least one
            value from each outer list.

        allow_duplicate : bool, default=False
            If True, allow duplicate use of tokens in include if tokens appear
            in multiple lists.

        Returns
        -------
        subtree : Optional[Span]
            Minimum subtree that includes at least one value from each includes
            outer list.
        """
    # Get all subtrees
    all_subtrees = list(subtrees(
        sentence = sentence,
        includes = includes,
        allow_duplicate = allow_duplicate,
    ))

    # Return None if no subtrees are found
    if len(all_subtrees) == 0:
        return None
    # Otherwise return minimum subtree
    else:
        return min(all_subtrees, key = lambda x: len(list(x.subtree)))
