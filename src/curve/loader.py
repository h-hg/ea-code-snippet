import os
from typing import Any

import numpy as np

from .. import loader


class CsvLoder(loader.Loader):
    def __init__(self, name: str, path_format: str, plot_opts: dict[Any, Any]):
        super().__init__(name)
        self.path_format = path_format
        self.plot_opts = plot_opts

    def load(self, **kw):
        path = self.path_format.format(**kw)
        if not os.path.isfile(path):
            # raise Exception(f"Error path of {path}")
            return None
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return {"mean": np.mean(data, axis=0), "std": np.std(data, axis=0)}
