################################################################################
import sys
from pathlib import Path
import math
import time

paths = [Path(__file__).absolute().parent, Path(__file__).absolute().parents[1]]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from jfi import jaxm

jaxm.set_default_dtype(jaxm.float64)

import objs

################################################################################

from sensitivity_jax.batch_sensitivity import implicit_hessian
from sensitivity_jax.sensitivity import implicit_hessian as implicit_hessian_
import objs

OPT = objs.CE()
X = jaxm.randn((100, 3))
Y = jaxm.randn((100, 5))
lam = 1e-3
blen = 2
p = jaxm.randn((blen, 3, 6))
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
    assert Dpz.shape == (blen,) + W.shape[1:] + p.shape[1:]
    assert Dppz.shape == (blen,) + W.shape[1:] + p.shape[1:] + p.shape[1:]

    Dpz2, Dppz2 = implicit_hessian(k_fn, W, p)
    Dpz2.shape == (blen,) + W.shape[1:] + p.shape[1:]
    Dppz2.shape == (blen,) + W.shape[1:] + p.shape[1:] + p.shape[1:]

    Dpz3, Dppz3 = [
        jaxm.stack(y) for y in zip(*[implicit_hessian_(k_fn, W_, p_) for (W_, p_) in zip(W, p)])
    ]
    Dppz3.shape == (blen,) + W.shape[1:] + p.shape[1:] + p.shape[1:]

    eps = max(jaxm.finfo(Dpz.dtype).resolution, 1e-9)

    err_Dpz2 = jaxm.norm(Dpz - Dpz2)
    err_Dppz2 = jaxm.norm(Dppz - Dppz2)
    err_Dpz3 = jaxm.norm(Dpz - Dpz3)
    err_Dppz3 = jaxm.norm(Dppz - Dppz3)

    if VERBOSE:
        print("err_Dpz: %9.4e" % err_Dpz2)
        print("err_Dppz: %9.4e" % err_Dppz2)
        print("err_Dpz: %9.4e" % err_Dpz3)
        print("err_Dppz: %9.4e" % err_Dppz3)

    assert err_Dpz2 < eps
    assert err_Dppz2 < eps
    assert err_Dpz3 < eps
    assert err_Dppz3 < eps


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

    Dpz2.shape == (blen,)
    assert Dppz2.shape == (blen,) + p.shape[1:]

    Dpz3, Dppz3 = [
        jaxm.stack(y)
        for y in zip(
            *[
                implicit_hessian_(k_fn, W_, p_, Dg=v_, jvp_vec=jvp_vec_)
                for (W_, p_, v_, jvp_vec_) in zip(W, p, v, jvp_vec)
            ]
        )
    ]
    assert Dppz3.shape == (blen,) + p.shape[1:]

    Dpz1 = jaxm.sum(
        Dpz1.reshape((blen, math.prod(p.shape[1:])))
        * jvp_vec.reshape((blen, math.prod(p.shape[1:]))),
        -1,
    ).reshape((blen,))
    Dppz1 = jaxm.sum(
        Dppz1.reshape((blen, math.prod(p.shape[1:]), math.prod(p.shape[1:])))
        * jvp_vec.reshape((blen, 1, math.prod(p.shape[1:]))),
        -1,
    ).reshape(p.shape)

    eps = max(jaxm.finfo(Dpz1.dtype).resolution, 1e-9)

    err_Dpz2 = jaxm.norm(Dpz1 - Dpz2)
    err_Dppz2 = jaxm.norm(Dppz1 - Dppz2)
    err_Dpz3 = jaxm.norm(Dpz1 - Dpz3)
    err_Dppz3 = jaxm.norm(Dppz1 - Dppz3)

    if VERBOSE:
        print("err_Dpz: %9.4e" % err_Dpz2)
        print("err_Dppz: %9.4e" % err_Dppz2)
        print("err_Dpz: %9.4e" % err_Dpz3)
        print("err_Dppz: %9.4e" % err_Dppz3)
    assert err_Dpz2 < eps
    assert err_Dppz2 < eps
    assert err_Dpz3 < eps
    assert err_Dppz3 < eps


def test_shape_jvp_with_Dzk_solve():
    Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
    test_shape_jvp_without_Dzk_solve(Dzk_solve_fn=Dzk_solve_fn)


if __name__ == "__main__":
    test_shape_and_val()
    test_shape_jvp_without_Dzk_solve()
    test_shape_jvp_with_Dzk_solve()
