import abc
import copy
import itertools
import os
from typing import Callable, List, Tuple

from .. import utils
from ..container import Container
from ..key_generator import KeyGenerator


class Metric(abc.ABC):

    @abc.abstractmethod
    def name() -> str:
        pass

    @abc.abstractmethod
    def set(self, table: "Table"):
        pass

    def set_cell(self, table: "Table", row: int, col: int, style: str | dict, **fmt_kw):
        if isinstance(style, str):
            table.data[row][col] = style.format(text=table.data[row][col], **fmt_kw)
        elif isinstance(style, dict):
            utils.deepin_update(table.attrs[row][col], style)


class Table(Container):
    def __init__(
        self,
        key_generator: KeyGenerator,
        cell_fmt: Callable[[dict], str] = lambda result: (
            "-" if not result else f"{result['mean']:.2E}±{result['std']:.2E}"
        ),
    ):
        super().__init__(key_generator)
        self.cell_fmt = cell_fmt
        self.metrics = []
        self.data = []
        self.attrs = []

    def clear(self):
        """
        clear the data.
        """
        self.data = []
        self.attrs = []

    def add_metrics(self, *metrics: Tuple[Metric]):
        l = []
        names = set()
        for metric in itertools.chain(reversed(metrics), reversed(self.metrics)):
            if metric.name not in names:
                l.append(metric)
                names.add(metric.name)
        self.metrics = list(reversed(l))

    def get_field_names(self) -> List[str]:
        return self.key_generator.names + [loader.name for loader in self.loaders]

    def _tbl_init(self):
        n_col = len(self.key_generator.names) + len(self.loaders)
        n_row = len(self.key_generator)
        self.data = [["" for _ in range(n_col)] for _ in range(n_row)]
        self.attrs = [[{} for _ in range(n_col)] for _ in range(n_row)]

    def _tbl_append_row(self):
        n_col = len(self.key_generator.names) + len(self.loaders)
        self.data.append(["" for _ in range(n_col)])
        self.attrs.append([{} for _ in range(n_col)])

    def __generate_data(self):
        """
        The table filed name is as follow:
            key_generator.name[0], ..., key_generator.name[-1], algo_results[0].name, ..., algo_results[-1].name
        """
        # generate the data of mean and std
        self._tbl_init()
        col_st = len(self.key_generator.names)
        for row, key in enumerate(self.get_key_generator()):
            self.data[row][:col_st] = list(key.values())
            for col, loader in enumerate(self.loaders, col_st):
                result = loader.get(**key)
                self.data[row][col] = self.cell_fmt(result)

        # clear the same key
        for col in range(len(self.key_generator.names)):
            last = self.data[0][col]
            for row in range(1, len(self.key_generator)):
                if self.data[row][col] == last:
                    self.data[row][col] = ""
                else:
                    last = self.data[row][col]

        for metric in self.metrics:
            metric.set(self)

    def __str__(self):
        if not self.data:
            self.__generate_data()

        from prettytable import PrettyTable

        t = PrettyTable()
        t.field_names = self.get_field_names()
        # make a deep copy
        data = copy.deepcopy(self.data)

        for row, line in enumerate(data):
            for col in range(len(line)):
                if "text" in self.attrs[row][col]:
                    style = self.attrs[row][col]["text"]
                    data[row][col] = style["fmt"].format(text=data[row][col], **style)

        t.add_rows(data)
        return str(t)

    def save(self, path: str, **kw):
        ext = os.path.splitext(os.path.basename(path))[1][1:]
        support_ext = {"csv", "xlsx"}
        if ext not in support_ext:
            return
        if not self.data:
            self.__generate_data()
        if ext == "csv":
            self.__save_as_csv(path)
        elif ext == "xlsx":
            self.__save_as_xlsx(path, kw.get("sheetname", "result"))

    def __save_as_csv(self, path: str):
        with open(path, mode="w", encoding="utf-8") as fw:
            fw.write(str(self))

    def __save_as_xlsx(self, path: str, sheetname: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(os.path.basename(path), exist_ok=True)

        import xlsxwriter

        # self.get_field_names()
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet(sheetname)
        for col, name in enumerate(self.get_field_names()):
            worksheet.write(0, col, name)

        # self.data
        for row, line in enumerate(self.data, 0):
            for col, elem in enumerate(line, 0):
                if "xlsx" in self.attrs[row][col]:
                    style = workbook.add_format(self.attrs[row][col]["xlsx"])
                    worksheet.write(row + 1, col, elem, style)
                else:
                    worksheet.write(row + 1, col, elem)
        workbook.close()
