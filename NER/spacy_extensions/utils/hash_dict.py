class HashableDict(dict):
    """Hashable dictionary to allow the dictionary to be used inside sets or as
        keys for other dictionaries. Changes adds a ``__hash__`` and ``__eq__``
        methods to a regular dictionary.
        
        Note
        ----
        This object is recursive, meaning that casting a dictionary of
        dictionaries will be transformed to a HashableDict of HashableDicts.
        """

    def __init__(self, *args, **kwargs):
        # Initialise normal dictionary
        super().__init__(*args, **kwargs)

        # Recursively cast nested dictionaries
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = HashableDict(value)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)