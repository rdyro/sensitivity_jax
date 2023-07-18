################################################################################
import sys
from pathlib import Path

paths = [Path(__file__).absolute().parent, Path(__file__).absolute().parents[1]]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from jfi import jaxm

jaxm.set_default_dtype(jaxm.float64)
################################################################################

from objs import LS, CE, OPT_conv

N = 100
X = jaxm.randn((N, 3, 6, 5)).reshape((N, -1))
Y = jaxm.randn((N, 10))

OPT = OPT_conv(LS(), 3, 5, 3, 2)
param = OPT.generate_parameter()
params = (param,)


def test_conv():
    W = OPT.solve(X, Y, *params)
    g = OPT.grad(W, X, Y, *params)
    assert jaxm.norm(g) < 1e-5

    eps = max(jaxm.finfo(g.dtype).resolution, 1e-9)

    # gradient quality
    g_ = jaxm.grad(OPT.fval)(W, X, Y, *params)
    err = jaxm.norm(g_ - g)
    assert err < eps

    # hessian quality
    H = OPT.hess(W, X, Y, *params)
    err = jaxm.norm(jaxm.jacobian(OPT.grad)(W, X, Y, *params) - H)
    assert err < eps

    # Dzk_solve
    H = OPT.hess(W, X, Y, *params).reshape((W.size, W.size))
    rhs = jaxm.randn((W.size, 3))
    err = jaxm.norm(jaxm.linalg.solve(H, rhs) - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=False))
    assert err < eps
    err = jaxm.norm(
        jaxm.linalg.solve(jaxm.t(H), rhs) - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=True)
    )
    assert err < eps


if __name__ == "__main__":
    test_conv()
