################################################################################
import sys
from pathlib import Path

paths = [Path(__file__).absolute().parent, Path(__file__).absolute().parents[1]]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from jfi import jaxm

jaxm.set_default_dtype(jaxm.float64)

import objs

################################################################################


X = jaxm.randn((100, 3))
Y = jaxm.randn((100, 5))


def generate_test(OPT, *params, name=""):
    def fn():
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
        err = jaxm.norm(
            jaxm.linalg.solve(H, rhs) - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=False)
        )
        assert err < eps
        err = jaxm.norm(
            jaxm.linalg.solve(jaxm.t(H), rhs) - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=True)
        )
        assert err < eps

    fn.__name__ = name
    return fn


names = ["LS", "CE", "LS_with_centers", "LS_with_diag", "CE_with_diag"]
OPTs = [
    objs.LS(),
    objs.CE(verbose=True, max_it=30),
    objs.OPT_with_centers(objs.LS(), 2),
    # objs.OPT_with_diag(objs.LS()),
    objs.LS_with_diag(),
    objs.OPT_with_diag(objs.CE(verbose=True, max_it=30)),
]
param_list = [
    (1e-1,),
    (1e-1,),
    (jaxm.array([1e-1, 1e-1]),),
    (1e-1 * jaxm.ones(X.shape[-1] * Y.shape[-1] + 1),),
    # (1e-1 * jaxm.ones(X.shape[-1]),),
    (1e-1 * jaxm.ones(X.shape[-1] * (Y.shape[-1] - 1)),),
]

for OPT, params, name in zip(OPTs, param_list, names):
    fn = generate_test(OPT, *params, name=name)
    setattr(sys.modules[__name__], f"test_{fn.__name__}", fn)
