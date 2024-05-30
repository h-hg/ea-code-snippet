import os
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np

from ..key_generator import KeyGenerator
from ..container import Container


class Curve(Container):
    def __init__(
        self,
        key_generator: KeyGenerator,
        n_yticks: int = 5,
        x_start: int = 5,
        x_step: int = 5,
        x_mark: int = 2,
        x_xticks_step: int = 2,
        zero_opt: Callable[[dict], float] | int = 1,
    ):
        """
        Note: Unlike Python, the index represented by the parameter starts from 1.
        It is more in line with the iteration of evolutionary algorithm,
        for the 0-th represents the initial population.

        n_yticks      The number of tick in y axis
        x_start       The generation that begins sampling. x_start >= 1
        x_step        The interval of sampling. x_step >= 1
        x_mark        The interval of marker. x_mark should be the times of x_step.
        x_xticks_step The interval of x ticks.
        zero_opt      The option of 0-th generation. If zero_opt is a function, it means that this function will be used to get the 0-th data.
                      If zero_opt is a integer i (it only be 0 or 1), it indicates that data from Loader starts from i-th generation.
                      The default value of opt is 1.
        """
        super().__init__(key_generator)
        self.n_yticks = n_yticks
        self.x_start = x_start
        self.x_step = x_step
        self.x_mark = x_mark
        self.x_xticks_step = x_xticks_step
        self.zero_opt = zero_opt

    def __get_log_base(self, data: List[dict | None]) -> float:
        data = list(filter(None, data))
        bd = min(np.min(item["mean"]) for item in data)
        bu = max(np.max(item["mean"]) for item in data)

        # bd * base^n_yticks = bu
        base = (bu / bd) ** (1 / self.n_yticks)
        base = round(base, 1)
        return base

    def save(self, path_fmt: str):
        for key in self.get_key_generator():
            data = [loader.get(**key) for loader in self.loaders]

            if all(item is None for item in data):
                continue

            # sampling
            for item in data:
                if item is None:
                    continue
                if self.zero_opt == 0:
                    item["y"] = item["mean"][self.x_start :: self.x_step]
                    item["y"] = np.insert(item["y"], 0, item["y"][0])
                elif self.zero_opt == 1:
                    item["y"] = item["mean"][self.x_start - 1 :: self.x_step]
                elif callable(self.zero_opt):
                    item["y"] = item["mean"][self.x_start - 1 :: self.x_step]
                    item["y"] = np.insert(item["y"], 0, self.zero_opt(**key))
                else:
                    raise Exception("Error opt")
                length = len(item["mean"])
            x = np.arange(1, length + 1)[self.x_start - 1 :: self.x_step]
            if self.zero_opt == 0 or callable(self.zero_opt):
                x = np.insert(x, 0, 0)

            # draw
            plt.figure()
            ax = plt.gca()
            ax.set_yscale("log", base=self.__get_log_base(data))
            for i, loader in enumerate(self.loaders):
                if data[i] is None:
                    continue
                y = data[i]["y"]
                plt.plot(
                    x,
                    y,
                    label=loader.name,
                    markevery=self.x_mark,
                    **loader.plot_opts,
                )
            plt.legend()
            plt.xlabel("Generation")
            plt.ylabel("Average fitness value")
            plt.xticks(x[:: self.x_xticks_step])
            path = path_fmt.format(**key)
            dirname = os.path.dirname(path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(path)
            # show
            # plt.show()
            plt.close()
