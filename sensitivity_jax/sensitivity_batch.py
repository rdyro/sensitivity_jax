import pdb
from typing import Callable, Mapping, Sequence, Union
from copy import copy

import numpy as np

# order of internal imports is important, jax_friendly_interface chooses a
# default device
from .jax_friendly_interface import init

jaxm = init()

# the order for the rest of the imports does not matter
from .utils import fn_with_sol_cache, prod
from .differentiation import JACOBIAN, HESSIAN, HESSIAN_DIAG
from .specialized_matrix_inverse import solve_gmres  # , solve_cg

from .sensitivity import implicit_jacobian as implicit_jacobian_

DeviceArray = jaxm.DeviceArray


def _generate_default_Dzk_solve_fn(
    optimizations: Mapping, k_fn: Callable
) -> Callable:
    """Generates the default Dzk (embedding Hessian) solution function A x = y

    Args:
        optimizations: dictionary with optional problem-specific optimizations
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function

    Returns:
        The `Dzk_solve_fn(z, *params, rhs=rhs, T=False)`, solving Dzk x = rhs.
    """

    def Dzk_solve_fn(z, *params, rhs=None, T=False):
        zlen = prod(z.shape[1:])
        if optimizations.get("Dzk", None) is None:
            optimizations["Dzk"] = JACOBIAN(
                lambda z, *params: jaxm.sum(k_fn(z, *params), 0)
            )(z, *params)
        Dzk = optimizations["Dzk"].reshape((zlen, -1, zlen)).swapaxes(0, 1)
        blen = Dzk.shape[0]
        if T:
            if optimizations.get("FT", None) is None:
                optimizations["FT"] = jaxm.linalg.lu_factor(jaxm.t(Dzk))
            FT = optimizations["FT"]
            return jaxm.linalg.lu_solve(FT, rhs.reshape((blen, zlen, -1)))
        else:
            if optimizations.get("F", None) is None:
                optimizations["F"] = jaxm.linalg.lu_factor(jaxm.t(Dzk))
            F = optimizations["F"]
            return jaxm.linalg.lu_solve(F, rhs.reshape((blen, zlen, -1)))

    optimizations["Dzk_solve_fn"] = Dzk_solve_fn


def _ensure_list(a):
    """Optionally convert a single element to a list. Leave a list unchanged."""
    return a if isinstance(a, (list, tuple)) else [a]


def implicit_jacobian2(
    k_fn: Callable,
    z: DeviceArray,
    *params: DeviceArray,
    Dg: DeviceArray = None,
    jvp_vec: Union[DeviceArray, Sequence[DeviceArray]] = None,
    matrix_free_inverse: bool = False,
    full_output: bool = False,
    optimizations: Mapping = None,
):
    return jaxm.vmap(
        lambda z, *params, Dg=None, jvp_vec=None: implicit_jacobian_(
            k_fn, z, *params, Dg=Dg, jvp_vec=jvp_vec
        )
    )(z, *params, Dg=Dg, jvp_vec=jvp_vec)


def implicit_jacobian(
    k_fn: Callable,
    z: DeviceArray,
    *params: DeviceArray,
    Dg: DeviceArray = None,
    jvp_vec: Union[DeviceArray, Sequence[DeviceArray]] = None,
    matrix_free_inverse: bool = False,
    full_output: bool = False,
    optimizations: Mapping = None,
):
    """Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec.

    Args:
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        z: the optimal embedding variable value array
        *params: the parameters p of the bilevel program
        Dg: left sensitivity vector (wrt z), for a VJP
        jvp_vec: right sensitivity vector(s) (wrt p) for a JVP
        matrix_free_inverse: whether to use approximate matrix inversion
        full_output: whether to append accumulated optimizations to the output
        optimizations: optional optimizations
    Returns:
        Jacobian/VJP/JVP as specified by arguments
    """

    optimizations = {} if optimizations is None else copy(optimizations)
    zlen, plen = prod(z.shape[1:]), [prod(param.shape[1:]) for param in params]
    blen = max(z.shape[0], max(param.shape[0] for param in params))

    z = jaxm.broadcast_to(z, (blen,) + z.shape[1:])
    params = [
        jaxm.broadcast_to(param, (blen,) + param.shape[1:]) for param in params
    ]

    if Dg is not None:
        Dg = jaxm.broadcast_to(Dg, (blen,) + Dg.shape[1:])

    if jvp_vec is not None:
        jvp_vec = _ensure_list(jvp_vec)
        jvp_vec = [
            jaxm.broadcast_to(vec, param.shape)
            for (vec, param) in zip(jvp_vec, params)
        ]

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        _generate_default_Dzk_solve_fn(optimizations, k_fn)
    #############################################################

    if Dg is not None:
        if matrix_free_inverse:
            raise NotImplementedError
            A_fn = lambda x: JACOBIAN(
                lambda z: jaxm.sum(k_fn(z, *params).reshape(-1) * x.reshape(-1))
            )(z).reshape(x.shape)
            v = -solve_gmres(A_fn, Dg.reshape((blen, zlen, 1)), max_it=300)
        else:
            Dzk_solve_fn = optimizations["Dzk_solve_fn"]
            v = -Dzk_solve_fn(
                z, *params, rhs=Dg.reshape((blen, zlen, 1)), T=True
            )
        fn = lambda *params: jaxm.sum(
            v.reshape((blen, zlen)) * k_fn(z, *params).reshape((blen, zlen))
        )
        Dp = JACOBIAN(fn, argnums=range(len(params)))(*params)
        Dp_shaped = [Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)]
        ret = Dp_shaped[0] if len(params) == 1 else Dp_shaped
    else:
        if jvp_vec is not None:
            fn = lambda *params: k_fn(z, *params)
            Dp = _ensure_list(jaxm.jvp(fn, tuple(params), tuple(jvp_vec))[1])
            Dp = [Dp.reshape((blen, zlen, 1)) for (Dp, plen) in zip(Dp, plen)]
            Dpk = Dp
        else:
            Dpk = JACOBIAN(
                lambda *params: jaxm.sum(k_fn(z, *params), 0),
                argnums=range(len(params)),
            )(*params)
            Dpk = [
                Dpk.reshape((zlen, blen, plen)).swapaxes(0, 1)
                for (Dpk, plen) in zip(Dpk, plen)
            ]

        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dpz = [-Dzk_solve_fn(z, *params, rhs=Dpk, T=False) for Dpk in Dpk]

        if jvp_vec is not None:
            Dpz_shaped = [Dpz.reshape(z.shape) for Dpz in Dpz]
        else:
            Dpz_shaped = [
                Dpz.reshape((blen,) + z.shape[1:] + param.shape[1:])
                for (Dpz, param) in zip(Dpz, params)
            ]
        ret = Dpz_shaped if len(params) != 1 else Dpz_shaped[0]
    return (ret, optimizations) if full_output else ret


def implicit_hessian(
    k_fn: Callable,
    z: DeviceArray,
    *params: DeviceArray,
    Dg: DeviceArray = None,
    Hg: DeviceArray = None,
    jvp_vec: Union[DeviceArray, Sequence[DeviceArray]] = None,
    optimizations: Mapping = None,
):
    """Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec.

    Args:
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        z: the optimal embedding variable value array
        *params: the parameters p of the bilevel program
        Dg: gradient sensitivity vector (wrt z), for chain rule
        Hg: Hessian sensitivity vector (wrt z), for chain rule
        jvp_vec: right sensitivity vector(s) (wrt p) for Hessian-vector-product
        optimizations: optional optimizations
    Returns:
        Hessian/chain rule Hessian as specified by arguments
    """
    optimizations = {} if optimizations is None else copy(optimizations)
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]
    if jvp_vec is not None:
        jvp_vec = _ensure_list(jvp_vec)
        jvp_vec = [
            jaxm.broadcast_to(vec, param.shape)
            for (vec, param) in zip(jvp_vec, params)
        ]
    if jvp_vec is not None:
        assert Dg is not None

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        _generate_default_Dzk_solve_fn(optimizations, k_fn)
    #############################################################

    # compute 2nd implicit gradients
    if Dg is not None:
        assert Dg.size == zlen
        assert Hg is None or Hg.size == zlen ** 2

        Dg_ = Dg.reshape((blen, zlen, 1))
        Hg_ = Hg.reshape((blen, zlen, zlen)) if Hg is not None else Hg

        # compute the left hand vector in the VJP
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        v = -Dzk_solve_fn(z, *params, rhs=Dg_.reshape((blen, zlen, 1)), T=True)
        fn = lambda z, *params: jaxm.sum(
            v.reshape((blen, zlen)) * k_fn(z, *params).reshape((blen, zlen))
        )

        if jvp_vec is not None:
            Dpz_jvp = _ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    jvp_vec=jvp_vec,
                    optimizations=optimizations,
                )
            )
            Dpz_jvp = [Dpz_jvp.reshape((blen, -1)) for Dpz_jvp in Dpz_jvp]

            # compute the 2nd order derivatives consisting of 4 terms
            # term 1 ##############################
            dfn_params = jaxm.grad(
                lambda *params: fn(z, *params), argnums=range(len(params))
            )
            Dpp1 = _ensure_list(
                jaxm.jvp(dfn_params, tuple(params), tuple(jvp_vec))[1]
            )
            Dpp1 = [
                Dpp1.reshape((blen, plen)) for (Dpp1, plen) in zip(Dpp1, plen)
            ]

            # term 2 ##############################
            Dpp2 = [
                jaxm.jvp(
                    lambda z: jaxm.grad(fn, argnums=i + 1)(z, *params),
                    (z,),
                    (Dpz_jvp.reshape(z.shape),),
                )[1].reshape((blen, plen))
                for (i, (Dpz_jvp, plen)) in enumerate(zip(Dpz_jvp, plen))
            ]

            # term 3 ##############################
            g_ = _ensure_list(
                jaxm.jvp(
                    lambda *params: jaxm.grad(fn)(z, *params),
                    tuple(params),
                    tuple(jvp_vec),
                )[1]
            )
            Dpp3 = [
                _ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        optimizations=optimizations,
                    )
                )[i].reshape((blen, plen))
                for (i, g_) in enumerate(g_)
            ]

            # term 4 ##############################
            g_ = [
                jaxm.jvp(
                    lambda z: jaxm.grad(fn)(z, *params),
                    (z,),
                    (Dpz_jvp.reshape(z.shape),),
                )[1]
                for (i, Dpz_jvp) in enumerate(Dpz_jvp)
            ]
            if Hg is not None:
                g_ = [
                    g_.reshape((blen, zlen))
                    + (Hg_ @ Dpz_jvp.reshape((blen, zlen, 1))).reshape(
                        (blen, zlen)
                    )
                    for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
                ]
            Dpp4 = [
                _ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        optimizations=optimizations,
                    )
                )[i].reshape((blen, plen))
                for ((i, g_), plen) in zip(enumerate(g_), plen)
            ]
            Dp = [
                jaxm.sum(
                    Dg_.reshape((blen, zlen)) * Dpz_jvp.reshape((blen, zlen)),
                    -1,
                )
                for Dpz_jvp in Dpz_jvp
            ]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [Dp.reshape((blen,)) for Dp in Dp]
            Dpp_shaped = [
                Dpp.reshape(param.shape) for (Dpp, param) in zip(Dpp, params)
            ]
        else:
            # compute the full first order gradients
            Dpz = _ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    optimizations=optimizations,
                )
            )
            Dpz = [
                Dpz.reshape((blen, zlen, plen))
                for (Dpz, plen) in zip(Dpz, plen)
            ]

            # compute the 2nd order derivatives consisting of 4 terms
            Dpp1 = HESSIAN_DIAG(lambda *params: fn(z, *params))(*params)
            # TODO here
            Dpp1 = [
                Dpp1.reshape((plen, plen)) for (Dpp1, plen) in zip(Dpp1, plen)
            ]

            temp = JACOBIAN(
                lambda *params: JACOBIAN(fn)(z, *params),
                argnums=range(len(params)),
            )(*params)
            temp = [
                jaxm.t(temp.reshape((zlen, plen)))
                for (temp, plen) in zip(temp, plen)
            ]
            Dpp2 = [
                (temp @ Dpz).reshape((plen, plen))
                for (temp, Dpz, plen) in zip(temp, Dpz, plen)
            ]
            Dpp3 = [jaxm.t(Dpp2) for Dpp2 in Dpp2]
            Dzz = HESSIAN(lambda z: fn(z, *params))(z).reshape((zlen, zlen))
            if Hg is not None:
                Dpp4 = [jaxm.t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            else:
                Dpp4 = [jaxm.t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
            Dp = [Dg_.reshape((1, zlen)) @ Dpz for Dpz in Dpz]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [
                Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)
            ]
            Dpp_shaped = [
                Dpp.reshape(param.shape + param.shape)
                for (Dpp, param) in zip(Dpp, params)
            ]
        return (
            (Dp_shaped[0], Dpp_shaped[0])
            if len(params) == 1
            else (Dp_shaped, Dpp_shaped)
        )
    else:
        Dpz, optimizations = implicit_jacobian(
            k_fn,
            z,
            *params,
            full_output=True,
            optimizations=optimizations,
        )
        Dpz = _ensure_list(Dpz)
        Dpz = [Dpz.reshape(zlen, plen) for (Dpz, plen) in zip(Dpz, plen)]

        # compute derivatives
        if optimizations.get("Dzzk", None) is None:
            Hk = HESSIAN_DIAG(k_fn)(z, *params)
            Dzzk, Dppk = Hk[0], Hk[1:]
            optimizations["Dzzk"] = Dzzk
        else:
            Dppk = HESSIAN_DIAG(lambda *params: k_fn(z, *params))(*params)
        Dzpk = JACOBIAN(
            lambda *params: JACOBIAN(k_fn)(z, *params),
            argnums=range(len(params)),
        )(*params)
        Dppk = [
            Dppk.reshape((zlen, plen, plen)) for (Dppk, plen) in zip(Dppk, plen)
        ]
        Dzzk = Dzzk.reshape((zlen, zlen, zlen))
        Dzpk = [
            Dzpk.reshape((zlen, zlen, plen)) for (Dzpk, plen) in zip(Dzpk, plen)
        ]
        Dpzk = [jaxm.t(Dzpk) for Dzpk in Dzpk]

        # solve the IFT equation
        lhs = [
            Dppk
            + Dpzk @ Dpz[None, ...]
            + jaxm.t(Dpz)[None, ...] @ Dzpk
            + (jaxm.t(Dpz)[None, ...] @ Dzzk) @ Dpz[None, ...]
            for (Dpz, Dzpk, Dpzk, Dppk) in zip(Dpz, Dzpk, Dpzk, Dppk)
        ]
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dppz = [
            -Dzk_solve_fn(
                z, *params, rhs=lhs.reshape((zlen, plen * plen)), T=False
            ).reshape((zlen, plen, plen))
            for (lhs, plen) in zip(lhs, plen)
        ]

        # return computed values
        Dpz_shaped = [
            Dpz.reshape(z.shape + param.shape)
            for (Dpz, param) in zip(Dpz, params)
        ]
        Dppz_shaped = [
            Dppz.reshape(z.shape + param.shape + param.shape)
            for (Dppz, param) in zip(Dppz, params)
        ]
        return (
            (Dpz_shaped[0], Dppz_shaped[0])
            if len(params) == 1
            else (Dpz_shaped, Dppz_shaped)
        )


def generate_optimization_fns(
    loss_fn: Callable,
    opt_fn: Callable,
    k_fn: Callable,
    normalize_grad: bool = False,
    optimizations: Mapping = None,
    jit: bool = True,
):
    """Directly generates upper/outer bilevel program derivative functions.

    Args:
        loss_fn: loss_fn(z, *params), upper/outer level loss
        opt_fn: opt_fn(*params) = z, lower/inner argmin function
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        normalize_grad: whether to normalize the gradient by its norm
        jit: whether to apply just-in-time (jit) compilation to the functions
    Returns:
        ``f_fn(*params), g_fn(*params), h_fn(*params)``
        parameters-only upper/outer level loss, gradient and Hessian.
    """
    sol_cache = {}
    optimizations = {} if optimizations is None else copy(optimizations)

    @fn_with_sol_cache(opt_fn, sol_cache, jit=jit)
    def f_fn(z, *params):
        return loss_fn(z, *params)

    @fn_with_sol_cache(opt_fn, sol_cache, jit=jit)
    def g_fn(z, *params):
        g = JACOBIAN(loss_fn, argnums=range(len(params) + 1))(z, *params)
        Dp = implicit_jacobian(
            k_fn, z, *params, Dg=g[0], optimizations=optimizations
        )
        Dp = Dp if len(params) != 1 else [Dp]
        ret = [Dp + g for (Dp, g) in zip(Dp, g[1:])]
        if normalize_grad:
            ret = [(z / (jaxm.norm(z) + 1e-7)) for z in ret]
        return ret[0] if len(ret) == 1 else ret

    @fn_with_sol_cache(opt_fn, sol_cache, jit=jit)
    def h_fn(z, *params):
        g = JACOBIAN(loss_fn, argnums=range(len(params) + 1))(z, *params)

        if optimizations.get("Hz_fn", None) is None:
            optimizations["Hz_fn"] = jaxm.hessian(loss_fn)
        Hz_fn = optimizations["Hz_fn"]
        Hz = Hz_fn(z, *params)
        H = [Hz] + HESSIAN_DIAG(lambda *params: loss_fn(z, *params))(*params)

        _, Dpp = implicit_hessian(
            k_fn,
            z,
            *params,
            Dg=g[0],
            Hg=H[0],
            optimizations=optimizations,
        )
        Dpp = Dpp if len(params) != 1 else [Dpp]
        ret = [Dpp + H for (Dpp, H) in zip(Dpp, H[1:])]
        return ret[0] if len(ret) == 1 else ret

    return (f_fn, g_fn, h_fn)
