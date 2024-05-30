import copy
import itertools
from typing import Tuple
from .loader import Loader
from .key_generator import KeyGenerator


class Container:
    def __init__(
        self,
        key_generator: KeyGenerator,
    ):
        self.key_generator = copy.deepcopy(key_generator)
        self.loaders = []

    def add_loaders(self, *loaders: Tuple[Loader]):
        l = []
        names = set()
        for loader in itertools.chain(reversed(loaders), reversed(self.loaders)):
            if loader.name not in names:
                l.append(loader)
                names.add(loader.name)
        self.loaders = list(reversed(l))

    def remove_loaders(self, *names: Tuple[str]):
        names = set(names)
        self.loaders = list(
            filter(lambda loader: loader.name not in names, self.loaders)
        )

    def get_key_generator(self) -> KeyGenerator:
        return copy.deepcopy(self.key_generator)
