from slope.solvers.admm_solver import admm
from slope.solvers.hybrid import hybrid_cd
from slope.solvers.newtalm import newt_alm
from slope.solvers.pgd import prox_grad
from slope.solvers.newtalm2 import newt_alm2
from slope.solvers.oracle import oracle_cd

__all__ = [
    "prox_grad",
    "hybrid_cd",
    "oracle_cd",
    "admm",
    "newt_alm",
    "newt_alm2"
]
