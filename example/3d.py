import os
import sys

workspace = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(workspace)

from src import landscape
from src import benchmark

f = benchmark.Sphere()
landscape.draw_3d(f, *f.bound(), 0.1, f"{f.name()}.pdf")
