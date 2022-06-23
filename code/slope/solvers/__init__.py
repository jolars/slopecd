from slope.solvers.pgd import prox_grad
from slope.solvers.hybrid import hybrid_cd
from slope.solvers.oracle import oracle_cd
from slope.solvers.admm_solver import admm

__all__ = ['prox_grad',
           'hybrid_cd',
           'oracle_cd',
           'admm']
