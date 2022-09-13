import numpy as np
from numba import njit


def cardcal(x, r, tiny):
    if "tiny" in dir():
        tiny = 1e-16
    n = len(x)
    normx1 = np.linalg.norm(x, ord=1)
    if min([normx1, np.linalg.norm(x, np.inf)]) <= tiny:
        k = 0
        return
    absx = sorted(abs(x), reverse=True)
    idx = list(np.argsort(-abs(x)))  # 取负号表示降序排列
    for i in range(n):
        if sum(absx[0:i]) >= r * normx1:
            k = i
            break
    return k
