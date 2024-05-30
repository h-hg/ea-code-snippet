import abc
from typing import Any


class Loader(abc.ABC):

    def __init__(self, name: str):
        """
        name The identifer of the loader
        """
        self.name = name
        self.data = {}

    def __hashKey(self, **kw) -> str:
        return str({key: kw[key] for key in sorted(kw.keys())})

    def name(self) -> str:
        return self.name

    def check(self, value: Any) -> bool:
        """
        Return True when value is validated.

        You can rewrite this method in the subclass
        """
        return bool(value)

    def get(self, **kw) -> Any:
        key = self.__hashKey(**kw)
        if key not in self.data:
            value = self.load(**kw)
            if not self.check(value):
                return None
            self.data[key] = value
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def load(self, **kw) -> Any:
        pass
