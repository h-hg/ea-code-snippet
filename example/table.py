import os
import sys

workspace = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(workspace)

from src import table
from src import KeyGenerator

key_generator = KeyGenerator(
    ["Benchmark", "Dimension"],
    ["ackley", "ellipsoid", "griewank", "qing", "rastrigin", "rosenbrock"],
    [100, 200, 300, 500, 1000],
)

# algorithm result
cc_ddea = table.CsvLoader(
    "CC-DDEA",
    workspace + "/example/data/CC-DDEA/cc-ddea-g=10_tg=8/{Benchmark}-{Dimension}d.csv",
)

cl_ddea = table.CsvLoader(
    "CL-DDEA",
    workspace + "/example/data/CL-DDEA/1000/{Benchmark}_d={Dimension}_y.csv",
)
ddea_se = table.CsvLoader(
    "DDEA-SE",
    workspace + "/example/data/DDEA-SE/1000/{Benchmark}_d={Dimension}_y.csv",
)
bddea_ldg = table.CsvLoader(
    "BDDEA-LDG",
    workspace + "/example/data/BDDEA-LDG/1000/{Benchmark}_d={Dimension}_y.csv",
)
srk_ddea = table.CsvLoader(
    "SRK-DDEA",
    workspace + "/example/data/SRK-DDEA/1000/{Benchmark}_d={Dimension}_y.csv",
)
tt_ddea = table.CsvLoader(
    "TT-DDEA",
    workspace + "/example/data/TT-DDEA/1000/{Benchmark}_d={Dimension}_y.csv",
)
ms_ddeo = table.CsvLoader(
    "MS-DDEO",
    workspace + "/example/data/MS-DDEO/1000/{Benchmark}_d={Dimension}_y.csv",
)


t1 = table.Table(key_generator)

t1.add_loaders(cc_ddea, cl_ddea, ddea_se, bddea_ldg, srk_ddea, tt_ddea, ms_ddeo)

binaryCmp = table.WilcoxonRankSumTest(cc_ddea)
# Latex settings
# binaryCmp.flags = ["$+$", "$\\approx$", "$-$"]

showBest = table.ShowBest()
# Latex settings
# showBest.text_style = "\\textbf{{{}}}"

averageRank = table.AverageRank()
t1.add_metrics(binaryCmp, averageRank, showBest)
print(t1)

t1.save("CC-DDEA-offline.xlsx", sheetname="1000")
