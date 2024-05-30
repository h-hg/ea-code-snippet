import abc
from typing import List
import sys

import numpy as np

from .table import Table, Metric
from ..loader import Loader


class PairwiseComparator(Metric):

    def __init__(
        self,
        main_loader: Loader,
        flags: List[str],
        text_fmt: str,
        tag: str,
    ):
        self.main_loader = main_loader
        self.flags = flags
        self.text_fmt = text_fmt
        self.tag = tag

    @abc.abstractmethod
    def compare(self, x: np.ndarray, y: np.ndarray):
        """
        Return:
            whether x is better than y
        Arg:
            alpha: confidence
        """
        pass

    def set(self, table: Table):
        row_st = 0
        col_st = len(table.key_generator.names)

        # sumaries = [better, approximate, worse]
        sumaries = [[0, 0, 0] for _ in range(len(table.loaders))]
        main_idx = -1
        for i, loader in enumerate(table.loaders):
            if loader.name == self.main_loader.name:
                main_idx = i
                break

        for row, key in enumerate(table.get_key_generator(), row_st):
            for col, loader in enumerate(table.loaders, col_st):
                if loader.name == self.main_loader.name:
                    continue
                y1, y2 = self.main_loader.get(**key).get("y", None), loader.get(
                    **key
                ).get("y", None)
                if y1 is None or y2 is None:
                    continue
                res = self.compare(y1, y2)
                if res == 1:
                    sumaries[col - col_st][0] += 1
                    flag = self.flags[0]
                elif res == 0:
                    sumaries[col - col_st][1] += 1
                    flag = self.flags[1]
                else:
                    sumaries[col - col_st][-1] += 1
                    flag = self.flags[-1]

                self.set_cell(table, row, col, self.text_fmt, flag=flag)

        table._tbl_append_row()

        for col, sumary in enumerate(sumaries, col_st):
            if col == main_idx + col_st:
                table.data[-1][col] = "NA"
            else:
                table.data[-1][col] = f"{self.tag}{sumary[0]}/{sumary[1]}/{sumary[2]}"
        table.data[-1][0] = "/".join(self.flags)


class WilcoxonRankSumTest(PairwiseComparator):

    def __init__(
        self,
        main_loader: Loader,
        alpha: float = 0.05,
        flags: List[str] = ["+", "≈", "-"],
        text_fmt: str = "{text}({flag})",
        tag: str = "",
    ):
        super().__init__(main_loader, flags, text_fmt, tag)
        self.alpha = alpha

    def name() -> str:
        return "Wilcoxon Rank Sum Test"

    def compare(self, x: np.ndarray, y: np.ndarray):
        from scipy import stats

        x = x.reshape(-1)
        y = y.reshape(-1)
        pvalue = stats.mannwhitneyu(x, y)[1]
        if pvalue < self.alpha:
            if np.average(x) < np.average(y):
                return 1  # "+"
            else:
                return -1  # "-"
        else:
            return 0  # "="


class WilcoxonSignedRankTest(PairwiseComparator):

    def __init__(
        self,
        main_loader: Loader,
        alpha: float = 0.05,
        flags: List[str] = ["+", "≈", "-"],
        text_fmt: str = "{text}({flag})",
        tag: str = "",
    ):
        super().__init__(main_loader, flags, text_fmt, tag)
        self.alpha = alpha

    def name() -> str:
        return "Wilcoxon Signed Sum Test"

    def compare(self, x: np.ndarray, y: np.ndarray):
        from scipy import stats

        x = x.reshape(-1)
        y = y.reshape(-1)
        pvalue = stats.wilcoxon(x, y)[1]
        if pvalue < self.alpha:
            if np.average(x) < np.average(y):
                return 1  # "+"
            else:
                return -1  # "-"
        else:
            return 0  # "="


class ShowBest(Metric):
    def __init__(
        self, style: dict = {"text": {"fmt": "<{text}>"}, "xlsx": {"bold": True}}
    ):
        self.style = style

    def name() -> str:
        return "Show Best"

    def set(self, table: Table):
        row_st = 0
        col_st = len(table.key_generator.names)
        for row, key in enumerate(table.get_key_generator(), row_st):
            min_val, min_idx = sys.float_info.max, -1
            for i, loader in enumerate(table.loaders):
                result = loader.get(**key).get("mean", None)
                # missing value
                if result == None:
                    continue
                if result < min_val:
                    min_val = result
                    min_idx = i
            if min_idx != -1:
                self.set_cell(table, row, col_st + min_idx, self.style)


class AverageRank(Metric):
    def __init__(self, text_fmt: str = "{rank:.2f}"):
        self.text_fmt = text_fmt

    def name() -> str:
        return "Average Rank"

    def __rank(self, score: List[float]) -> List[int]:
        # missing value
        x = list(filter(lambda val: val != None, score))
        x = sorted(list(set(x)))
        # +1: rank starts from 1
        x = dict(zip(x, range(1, len(x) + 1)))
        x[None] = 0
        return [x[val] for val in score]

    def set(self, table: Table):
        row_st = 0
        col_st = len(table.key_generator.names)
        key_generator = table.get_key_generator()
        sum_of_ranks = np.zeros(len(table.loaders))
        count = np.ones_like(sum_of_ranks) * len(key_generator)
        for key in key_generator:
            means = [loader.get(**key).get("mean", None) for loader in table.loaders]
            ranks = np.array(self.__rank(means))
            not_join_in = np.where(ranks == 0, 1, 0)
            count -= not_join_in
            sum_of_ranks += ranks
        # average_ranks = sum_of_ranks / np.array([len(algo_result) for algo_result in algo_results])
        average_ranks = sum_of_ranks / count

        table._tbl_append_row()
        for col, average_rank in enumerate(average_ranks, col_st):
            table.data[-1][col] = self.text_fmt.format(rank=average_rank)
        table.data[-1][0] = "Average Rank"
