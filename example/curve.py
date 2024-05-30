import os
import sys

workspace = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(workspace)

from src import benchmark
from src import curve
from src import KeyGenerator
from src import sampling
from src import Loader

loaders = [
    curve.CsvLoder(
        "CC-DDEA",
        workspace + "/example/data/CC-DDEA/cc-ddea-g=10_tg=8/{f_name}-{d}d.csv",
        {"linestyle": "solid", "marker": "s"},
    ),
    curve.CsvLoder(
        "CL-DDEA",
        workspace + "/example/data/CL-DDEA/1000/{f_name}_d={d}_y.csv",
        {"linestyle": "dashdot", "marker": "<"},
    ),
    curve.CsvLoder(
        "SRK-DDEA",
        workspace + "/example/data/SRK-DDEA/1000/{f_name}_d={d}_y.csv",
        {"linestyle": "dotted", "marker": "o"},
    ),
    curve.CsvLoder(
        "TT-DDEA",
        workspace + "/example/data/TT-DDEA/1000/{f_name}_d={d}_y.csv",
        {"linestyle": "dashed", "marker": "v"},
    ),
    curve.CsvLoder(
        "DDEA-SE",
        workspace + "/example/data/DDEA-SE/1000/{f_name}_d={d}_y.csv",
        {"linestyle": "dashdot", "marker": "^"},
    ),
    curve.CsvLoder(
        "BDDEA-LDG",
        workspace + "/example/data/BDDEA-LDG/1000/{f_name}_d={d}_y.csv",
        {"linestyle": (5, (10, 3)), "marker": "*"},
    ),
    curve.CsvLoder(
        "MS-DDEO",
        workspace + "/example/data/MS-DDEO/1000/{f_name}_d={d}_y.csv",
        {"linestyle": (0, (5, 10)), "marker": "x"},
    ),
]

key_generator = KeyGenerator(
    ["f_name", "d"],
    ["ackley", "ellipsoid", "griewank", "qing", "rastrigin", "rosenbrock"],
    [100, 200, 300, 500, 1000],
)


class LHSMin(Loader):
    def __init__(self):
        super().__init__("LHSMin")

    def load(self, **kw):
        import inspect
        import numpy as np

        for cls in [obj for obj in vars(benchmark).values() if inspect.isclass(obj)]:
            module_name = inspect.getmodule(cls).__name__
            if cls == benchmark.Benchmark or module_name != benchmark.__name__:
                continue
            f = cls()
            if f.name().lower() != kw["f_name"].lower():
                continue
            d = kw["d"]
            n = 1000
            x = sampling.lhs_np(n, d, *f.bound())
            y = f(x)
            return np.min(y)

    def __call__(self, **kw):
        return self.get(**kw)


fig = curve.Curve(key_generator, zero_opt=LHSMin())
fig.add_loaders(*loaders)
fig.save("./curve/{f_name}_{d}.pdf")
