################################################################################
import sys
from pathlib import Path
import time

paths = [Path(__file__).absolute().parent, Path(__file__).absolute().parents[1]]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from jfi import jaxm

jaxm.set_default_dtype(jaxm.float64)

import jax.numpy as jnp

################################################################################

from sensitivity_jax.sensitivity import implicit_hessian
import objs

OPT = objs.CE()
X = jaxm.randn((100, 3))
Y = jaxm.randn((100, 5))
lam = 1e-3
p = jaxm.randn((3, 6))
W = OPT.solve(X @ p, Y, lam)

VERBOSE = True


# we test here 2nd order implicit gradients
def test_shape_and_val():
    if VERBOSE:
        print()
    k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
    Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
    optimizations = dict(Dzk_solve_fn=Dzk_solve_fn)
    Dpz, Dppz = implicit_hessian(k_fn, W, p, optimizations=optimizations)
    assert Dpz.shape == (W.shape + p.shape)
    assert Dppz.shape == (W.shape + p.shape + p.shape)

    Dpz2, Dppz2 = implicit_hessian(k_fn, W, p)
    assert Dpz2.shape == (W.shape + p.shape)
    assert Dppz2.shape == (W.shape + p.shape + p.shape)

    eps = 1e-5
    err_Dpz = jaxm.norm(Dpz - Dpz2)
    err_Dppz = jaxm.norm(Dppz - Dppz2)

    if VERBOSE:
        print("err_Dpz: %9.4e" % err_Dpz)
        print("err_Dppz: %9.4e" % err_Dppz)

    assert err_Dpz < eps
    assert err_Dppz < eps


def test_shape_jvp_without_Dzk_solve(Dzk_solve_fn=None):
    if VERBOSE:
        print()

    k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
    jvp_vec = jaxm.randn(p.shape)
    v = jaxm.randn(W.shape)

    t_ = time.time()
    optimizations = dict(Dzk_solve_fn=Dzk_solve_fn)
    Dpz1, Dppz1 = implicit_hessian(k_fn, W, p, Dg=v, optimizations=optimizations)
    if VERBOSE:
        print("Elapsed %9.4e" % (time.time() - t_))

    t_ = time.time()
    Dpz2, Dppz2 = implicit_hessian(k_fn, W, p, Dg=v, jvp_vec=jvp_vec, optimizations=optimizations)
    if VERBOSE:
        print("Elapsed %9.4e" % (time.time() - t_))

    assert Dpz2.shape == ()
    assert Dppz2.shape == p.shape

    eps = 1e-5
    err_Dpz = jnp.linalg.norm(Dpz1.reshape(p.size) @ jvp_vec.reshape(p.size) - Dpz2)
    err_Dppz = jnp.linalg.norm(
        Dppz1.reshape((p.size, p.size)) @ jvp_vec.reshape(p.size) - Dppz2.reshape(p.size)
    )
    if VERBOSE:
        print("err_Dpz: %9.4e" % err_Dpz)
        print("err_Dppz: %9.4e" % err_Dppz)
    assert err_Dpz < eps
    assert err_Dppz < eps

def test_shape_jvp_with_Dzk_solve():
    Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
    test_shape_jvp_without_Dzk_solve(Dzk_solve_fn=Dzk_solve_fn)

if __name__ == "__main__":
    test_shape_and_val()
    test_shape_jvp_with_Dzk_solve()
    test_shape_jvp_without_Dzk_solve()