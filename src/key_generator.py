import itertools
from typing import List, Tuple, Any


class KeyGenerator:
    def __init__(self, names: List[str], *data: Tuple[List[Any]]):
        """
        names: A list composed of names from various dimension
        data: A list composed of data that needs to be enumerated for each dimension
        """
        if len(names) != len(data):
            raise Exception("The size of names is not equal to data")
        self.names = names
        self.iter = itertools.product(*data)

        self.n = 1
        for val in data:
            self.n *= len(val)
        if self.n == 1:
            self.n = 0

    def dimension(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        return dict(zip(self.names, next(self.iter)))
