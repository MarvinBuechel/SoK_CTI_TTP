class Trie(object):
    """Trie implements a Trie data structure that performs quick lookups of
        strings in text.
        """

    def __init__(self, ignore_case=False, final_node='__final__'):
        """Trie object for fast detection of multiple sequences simultaneously.

            Parameters
            ----------
            ignore_case : boolean, default=False
                If True, ignore cases in added sequences and text to predict.

            final_node : hashable object, default='__final__'
                Object used to indicate that a node is final. Note that this
                cannot be used as a transition. As transition are often single
                characters, when using a string, please use multiple characters.
            """
        # Initialise graph
        self.graph = dict()
        # Set flag to ignore case
        self.ignore_case = ignore_case
        # Set final node name
        self.final_node = final_node

        # Set entries
        self.entries = list()

        # Store depth of graph
        self.depth = 0

    ########################################################################
    #                         Fit/Predict methods                          #
    ########################################################################

    def fit(self, sequences, labels):
        """Add sequences and labels to lookup function.

            Parameters
            ----------
            sequences : iterable of string, shape=(n_sequences,)
                String representations of sequences to add.
                Note that each character is treated as a transition, there is no
                support for special regex characters.

            labels : iterable of object, shape=(n_sequences,)
                Label corresponding to each sequence.

            Returns
            -------
            self : self
                Returns self
            """
        # Reset depth
        self.depth = 0

        # Add individual sequences
        for sequence, label in zip(sequences, labels):
            self.fit_single(sequence, label)

        # Return self
        return self


    def fit_single(self, sequence, label):
        """Add a sequence and label to Trie.

            Parameters
            ----------
            sequence : string
                String representation of sequence to add.
                Note that each character is treated as a transition, there is no
                support for special regex characters.

            label : object
                Object that labels the sequence.
            """
        # Add entry
        self.entries.append((sequence, label))

        # Check if case should be ignored
        if self.ignore_case:
            sequence = sequence.lower()

        # Get starting node
        node = self.graph

        # Loop over characters in sequence
        for character in sequence:
            # Add transition if required
            if not character in node:
                node[character] = dict()
            # Make transition
            node = node[character]

        # Add final node as label
        if self.final_node not in node:
            node[self.final_node] = set()
        node[self.final_node].add(label)

        # Set depth
        self.depth = max(self.depth, len(sequence))

        # Return self
        return self


    def predict(self, string):
        """Perform detection of sequences in text.

            Parameters
            ----------
            string : string
                Text in which to detect sequences.

            Returns
            -------
            result : set of tuples
                - span_start : int, start index of detected IoC span.
                - span_end   : int, start index of detected IoC span.
                - label      : object, label corresponding to detected IoC.
            """
        # Initialise result
        result = set()

        # Loop over all substrings
        for i in range(len(string)):
            # Gather partial results
            # Important speedup: Only copy string from start (i) until maximum
            # known Trie depth (i + self.depth + 1). This reduces memory copies
            # for large strings.
            partial_result = self.process(string[i : i+self.depth+1])

            if len(partial_result) == 1:
                # Loop over output
                for span_end, label in partial_result.items():
                    # Add result as tuple
                    result.add((i, i+span_end+1, tuple(sorted(label))))
            elif len(partial_result) > 1:
                # Perform greedy search
                span_end = max(partial_result)
                label = partial_result[span_end]
                # Add result as tuple
                result.add((i, i+span_end+1, tuple(sorted(label))))

        # Return result
        return result

    ########################################################################
    #                           Process strings                            #
    ########################################################################

    def process(self, string):
        """Process single string only from the start based on parsed tokens.

            Parameters
            ----------
            string : string
                String to parse for tokens.

            Returns
            -------
            result : dict()
                Dictionary of end_index -> set of tokens.
            """
        # Initialise result
        result = dict()

        # Get starting node
        node = self.graph

        # Check if case should be ignored
        if self.ignore_case:
            string = string.lower()

        # Perform each transition in string
        for index, character in enumerate(string):

            # Perform transition
            node = node.get(character)

            # Check if transition was unsuccessful
            if node is None:
                break

            # If we reached a possible final node, add to result
            elif node.get(self.final_node):
                result[index] = node.get(self.final_node)

        # Return result
        return result
