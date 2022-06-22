from slope.solvers.hybrid import hybrid_cd
from slope.solvers.oracle import oracle_cd
from slope.solvers.pgd import prox_grad
from slope.solvers.newtalm import newt_alm

__all__ = ["prox_grad", "hybrid_cd", "oracle_cd", "newt_alm"]
