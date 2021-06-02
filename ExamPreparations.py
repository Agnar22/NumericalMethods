import numpy as np
import scipy.linalg
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


def lu_factorization(mat: np.matrix, permute: bool) -> None:
    print("LU factorization:")
    if not permute:
        p, l, u = scipy.linalg.lu(mat, permute_l=permute)
        print("p", p)
        print("l", l)
        print("u", u)
    else:
        pl, u = scipy.linalg.lu(mat, permute_l=permute)
        print("pl", pl)
        print("u", u)
    print()


def qr_factorization(mat: np.matrix) -> None:
    print("QR factorization:")
    q, r = scipy.linalg.qr(mat)
    print("q", q)
    print("r", r)
    print()


def svd(mat: np.matrix) -> None:
    print("SVD")
    u, s, vh = scipy.linalg.svd(mat)
    print("U", u)
    print("s", s)
    print("v", vh)
    print()


def householder_transformation(mat: np.array) -> None:
    ...


def normal_equations(A: np.matrix, b: np.array) -> None:
    print("Normal equations")
    x = scipy.linalg.inv(A.T @ A) @ A.T @ b
    print("x", x)


# def runge_kutta_methods(func: Callable[[np.longdouble], np.longdouble],
#                         start: np.float,
#                         end: np.float,
#                         A: np.ndarray[np.float],
#                         b: np.ndarray[np.float],
#                         c: np.ndarray[np.float],
#                         N: np.int) -> np.longdouble:
#     print(np.allclose(A, np.tril(A)))
#     y = start
#     for n in range(N+1):


if __name__ == '__main__':
    print(derivative(lambda x: x + 2, np.longdouble(3)))

    fixed_point_iteration(lambda x: np.sin(x) + 1, np.longdouble(np.pi - 0.2), 30)
    newton_iteration(lambda x: np.sin(x) + 1, lambda x: np.cos(x), np.longdouble(np.pi - 0.2), 10)
    newton_iteration(lambda x: np.sin(x) + 1, None, np.longdouble(np.pi - 0.2), 10)

    # Numerical Methods for ODEs.
    #  - Forward Euler
    A, b, c = np.matrix([[0]]), np.array([1]), np.array([0])
    #  - Backward Euler
    # A, b, c = np.matrix([[1]]), np.array([1]), np.array([1])
    #  - Heuns method
    # A, b, c = np.matrix([[0, 0], [1, 0]]), np.array([0.5, 0.5]), np.array([0, 1])
    #  - Midpoint method
    # A, b, c = np.matrix([[0.5]]), np.array([0.5, 0.5]), np.array([1])
    #  - Improved Euler
    # A, b, c = np.matrix([[0,0], [0.5, 0]), np.array([0, 1]), np.array([0, 0.5])

    mat = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    lu_factorization(mat, True)
    lu_factorization(mat, False)
    qr_factorization(mat)
    svd(mat)

    A = np.mat([[3, 1], [1, 1], [4, 2]])
    b = np.array([1, 0, 2])
    normal_equations(A, b)
