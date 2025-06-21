"""Iterator utilities (itertools extensions) used by library."""
from typing import Any, Iterable, Iterator, List, Tuple


def iterable_combinations(
        iterables: List[Iterable[Any]],
    ) -> Iterator[Tuple[Any]]:
    """Loop over all combinations of values for given iterables.
    
        Parameters
        ----------
        iterables : List[Iterable[Any]]
            Iterables for which to get combinations.
            
        Yields
        ------
        combination : Tuple[Any]
            Tuple of values from given iterables.
        """
    # Pop initial value from iterables
    initial = iterables[0]

    # Yield from initial if there are no more iterables
    if len(iterables) == 1:
        for value in initial:
            yield (value,)

    # Otherwise yield from combinations
    else:
        for value in initial:
            for combination in iterable_combinations(iterables[1:]):
                yield (value, *combination)