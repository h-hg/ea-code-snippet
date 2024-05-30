import os
from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt


def draw_3d(
    objective_func: Callable[[np.ndarray], float],
    lower_bound: float | Tuple[float, float],
    upper_bound: float | Tuple[float, float],
    step: float | Tuple[float, float] = 0.1,
    path: str = "",
):
    """
    objective_func : Objective function
    lower_bound : The lower bound of decision variables.
                 If it is a float, it means that the lower limits of the two variables are the same
    upper_bound : It is similar to lower_bound
    step : Sampling interval
    """
    if np.size(lower_bound) == 1:
        lower_bound = (lower_bound, lower_bound)
    if np.size(upper_bound) == 1:
        upper_bound = (upper_bound, upper_bound)
    if np.size(step) == 1:
        step = (step, step)

    x1 = np.arange(lower_bound[0], upper_bound[0], step[0])
    len1 = len(x1)
    x2 = np.arange(lower_bound[1], upper_bound[1], step[1])
    len2 = len(x2)

    X1, X2 = np.meshgrid(x1, x2)
    x = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])
    Z = objective_func(x).reshape(len1, len2)

    # settings
    plt.rcParams["text.usetex"] = True
    # Plot the landscape
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, Z, cmap="viridis")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$z$")
    if path:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(path)
    plt.show()
    plt.close()
