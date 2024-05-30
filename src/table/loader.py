import os
from typing import Any

import numpy as np

from ..loader import Loader


class CsvLoader(Loader):
    def __init__(self, name: str, path_format: str):
        super().__init__(name)
        self.path_format = path_format

    def check(self, value: Any) -> bool:
        return True

    def load(self, **kw):
        file_path = self.path_format.format(**kw)
        if not os.path.isfile(file_path):
            # raise Exception(f"There is no file '{file_path}'")
            return {}
        y = np.loadtxt(file_path)
        # missing value
        if y is None:
            # raise Exception(f"Error format of '{file_path}'")
            return {}
        if y.ndim == 1:
            y = y.reshape(1, -1)
        y = y[:, -1].reshape(-1)
        return {
            "y": y,
            "mean": np.mean(y),
            "std": np.std(y),
        }
