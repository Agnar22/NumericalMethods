import numpy as np
from typing import Callable


def derivative(func: Callable[[np.longdouble], np.longdouble], x: np.longdouble) -> np.longdouble:
    h: np.longdouble = 1e-12
    return (func(x + h) - func(x)) / h


def fixed_point_iteration(func: Callable[[np.longdouble], np.longdouble], x: np.longdouble,
                          steps: np.int) -> np.longdouble:
    print("Fixed point iteration:")
    curr_x = x
    for step in range(steps):
        print(step, curr_x)
        curr_x = func(curr_x)
    print()


def newton_iteration(func: Callable[[np.longdouble], np.longdouble],
                     der: Callable[[np.longdouble], np.longdouble],
                     x: np.longdouble,
                     steps: np.int) -> np.longdouble:
    print("Newton iteration:")
    curr_x = x
    for step in range(steps):
        print(step, curr_x)
        if der is not None:
            curr_x = curr_x - func(curr_x) / der(curr_x)
        else:
            curr_x = curr_x - func(curr_x) / derivative(func, curr_x)
    print()


if __name__ == '__main__':
    print(derivative(lambda x: x + 2, np.longdouble(3)))

    fixed_point_iteration(lambda x: np.sin(x) + 1, np.longdouble(np.pi - 0.2), 30)
    newton_iteration(lambda x: np.sin(x) + 1, lambda x: np.cos(x), np.longdouble(np.pi - 0.2), 30)
    newton_iteration(lambda x: np.sin(x) + 1, None, np.longdouble(np.pi - 0.2), 30)
