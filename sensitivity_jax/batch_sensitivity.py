import pdb
from typing import Callable, Mapping, Sequence, Union
from copy import copy

import numpy as np

from jfi import jaxm

# the order for the rest of the imports does not matter
from .utils import fn_with_sol_cache, prod, bmv
from .differentiation import JACOBIAN, HESSIAN, HESSIAN_DIAG
from .differentiation import BATCH_JACOBIAN, BATCH_HESSIAN, BATCH_HESSIAN_DIAG
from .specialized_matrix_inverse import solve_gmres  # , solve_cg

from .sensitivity import implicit_jacobian as implicit_jacobian_
from .sensitivity import implicit_hessian as implicit_hessian_

JAXArray = jaxm.jax.Array


def _ensure_list(a):
    """Optionally convert a single element to a list. Leave a list unchanged."""
    return a if isinstance(a, (list, tuple)) else [a]


def _split_for_broadcast(d: Mapping, M: int):
    if d is None:
        return None
    cond = (
        # lambda v: hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == M
        lambda v: len(jaxm.shape(v)) >= 1
        and jaxm.shape(v)[0] == M
    )
    d_broadcast = {k: v for (k, v) in d.items() if cond(v)}
    d_other = {k: v for (k, v) in d.items() if not cond(v)}
    return d_broadcast, d_other


def _split_into_array_and_not(d: Mapping):
    """Split a dictionary into an array and other. Does not work with JAX
    Tracers."""
    if d is None:
        return None
    d_array = {
        k: v for (k, v) in d.items() if isinstance(v, (JAXArray, np.ndarray))
    }
    d_other = {
        k: v
        for (k, v) in d.items()
        if not isinstance(v, (JAXArray, np.ndarray))
    }
    return d_array, d_other


def _generate_default_Dzk_solve_fn(optimizations: Mapping, k_fn: Callable):
    """Generates the default Dzk (embedding Hessian) solution function A x = y

    Args:
        optimizations: dictionary with optional problem-specific optimizations
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function

    Returns:
        The `Dzk_solve_fn(z, *params, rhs=rhs, T=False)`, solving Dzk x = rhs.
    """

    def Dzk_solve_fn(z, *params, rhs=None, T=False):
        blen, zlen = z.shape[0], prod(z.shape[1:])
        if optimizations.get("Dzk", None) is None:
            optimizations["Dzk"] = BATCH_JACOBIAN(k_fn)(z, *params)
        Dzk = optimizations["Dzk"]
        if T:
            if optimizations.get("FT", None) is None:
                optimizations["FT"] = jaxm.linalg.lu_factor(
                    jaxm.t(Dzk.reshape((blen, zlen, zlen)))
                )
            return jaxm.linalg.lu_solve(optimizations["FT"], rhs)
        else:
            if optimizations.get("F", None) is None:
                optimizations["F"] = jaxm.linalg.lu_factor(
                    Dzk.reshape((blen, zlen, zlen))
                )
            return jaxm.linalg.lu_solve(optimizations["F"], rhs)

    optimizations["Dzk_solve_fn"] = Dzk_solve_fn


################################################################################


def implicit_jacobian2(
    k_fn: Callable,
    z: JAXArray,
    *params: JAXArray,
    nondiff_kw: Mapping = None,
    Dg: JAXArray = None,
    jvp_vec: Union[JAXArray, Sequence[JAXArray]] = None,
    matrix_free_inverse: bool = False,
    full_output: bool = False,
    optimizations: Mapping = None,
):
    """Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec,
    using vmap.

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

    # we need a custom rule for optimizations
    optimizations = {} if optimizations is None else optimizations
    nondiff_kw = {} if nondiff_kw is None else nondiff_kw
    # optimizations = _split_into_array_and_not(optimizations)
    # nondiff_kw = _split_into_array_and_not(nondiff_kw)
    optimizations = _split_for_broadcast(optimizations, z.shape[0])
    nondiff_kw = _split_for_broadcast(nondiff_kw, z.shape[0])

    # call the function with vmap
    return jaxm.vmap(
        lambda z, *params, nondiff_kw_=None, Dg=None, jvp_vec=None, optimizations_=None: implicit_jacobian_(
            k_fn,
            z,
            *params,
            nondiff_kw=None
            if nondiff_kw is None
            else dict(nondiff_kw_, **nondiff_kw[1]),
            Dg=Dg,
            jvp_vec=jvp_vec,
            optimizations=None
            if optimizations is None
            else dict(optimizations_, **optimizations[1]),
        )
    )(
        z,
        *params,
        nondiff_kw_=None if nondiff_kw is None else nondiff_kw[0],
        Dg=Dg,
        jvp_vec=jvp_vec,
        optimizations_=None if optimizations is None else optimizations[0],
    )


def implicit_jacobian(
    k_fn: Callable,
    z: JAXArray,
    *params: JAXArray,
    nondiff_kw: Mapping = None,
    Dg: JAXArray = None,
    jvp_vec: Union[JAXArray, Sequence[JAXArray]] = None,
    matrix_free_inverse: bool = False,
    full_output: bool = False,
    optimizations: Mapping = None,
):
    """Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec,
    using batched operations.

    Args:
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        z: the optimal embedding variable value array
        *params: the parameters p of the bilevel program
        nondiff_kw: nondifferentiable parameters to the implicit function
        Dg: left sensitivity vector (wrt z), for a VJP
        jvp_vec: right sensitivity vector(s) (wrt p) for a JVP
        matrix_free_inverse: whether to use approximate matrix inversion
        full_output: whether to append accumulated optimizations to the output
        optimizations: optional optimizations
    Returns:
        Jacobian/VJP/JVP as specified by arguments
    """

    optimizations = {} if optimizations is None else copy(optimizations)
    blen, zlen = z.shape[0], prod(z.shape[1:])
    plen = [prod(param.shape[1:]) for param in params]

    jvp_vec = _ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec

    if nondiff_kw is not None:
        k_fn_ = lambda z, *params: k_fn(z, *params, **nondiff_kw)
    else:
        k_fn_ = k_fn

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        _generate_default_Dzk_solve_fn(optimizations, k_fn_)
    #############################################################

    if Dg is not None:
        if matrix_free_inverse:
            raise NotImplementedError
        else:
            Dzk_solve_fn = optimizations["Dzk_solve_fn"]
            v = -Dzk_solve_fn(
                z, *params, rhs=Dg.reshape((blen, zlen, 1)), T=True
            )
        fn = lambda *params: jaxm.sum(
            v.reshape((blen, zlen)) * k_fn_(z, *params).reshape((blen, zlen))
        )
        Dp = JACOBIAN(fn, argnums=range(len(params)))(*params)
        Dp_shaped = [Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)]
        ret = Dp_shaped[0] if len(params) == 1 else Dp_shaped
    else:
        if jvp_vec is not None:
            fn = lambda *params: k_fn_(z, *params)
            Dp = _ensure_list(jaxm.jvp(fn, params, tuple(jvp_vec))[1])
            Dp = [Dp.reshape((blen, zlen, 1)) for (Dp, plen) in zip(Dp, plen)]
            Dpk = Dp
        else:
            Dpk = BATCH_JACOBIAN(
                lambda *params: k_fn_(z, *params),
                argnums=range(len(params)),
            )(*params)
            Dpk = [
                Dpk.reshape((blen, zlen, plen))
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


################################################################################


def implicit_hessian2(
    k_fn: Callable,
    z: JAXArray,
    *params: JAXArray,
    nondiff_kw: Mapping = None,
    Dg: JAXArray = None,
    Hg: JAXArray = None,
    jvp_vec: Union[JAXArray, Sequence[JAXArray]] = None,
    optimizations: Mapping = None,
):
    """Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec,
    using vmap.

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
    # we need a custom rule for optimizations
    optimizations = {} if optimizations is None else optimizations
    nondiff_kw = {} if nondiff_kw is None else nondiff_kw
    optimizations = _split_for_broadcast(optimizations, z.shape[0])
    nondiff_kw = _split_for_broadcast(nondiff_kw, z.shape[0])

    # call the function with vmap
    return jaxm.vmap(
        lambda z, *params, nondiff_kw_=None, Dg=None, Hg=None, jvp_vec=None, optimizations_=None: implicit_hessian_(
            k_fn,
            z,
            *params,
            nondiff_kw=None
            if nondiff_kw is None
            else dict(nondiff_kw_, **nondiff_kw[1]),
            Dg=Dg,
            Hg=Hg,
            jvp_vec=jvp_vec,
            optimizations=None
            if optimizations is None
            else dict(optimizations_, **optimizations[1]),
        )
    )(
        z,
        *params,
        nondiff_kw_=None if nondiff_kw is None else nondiff_kw[0],
        Dg=Dg,
        Hg=Hg,
        jvp_vec=jvp_vec,
        optimizations_=None if optimizations is None else optimizations[0],
    )


def implicit_hessian(
    k_fn: Callable,
    z: JAXArray,
    *params: JAXArray,
    nondiff_kw: Mapping = None,
    Dg: JAXArray = None,
    Hg: JAXArray = None,
    jvp_vec: Union[JAXArray, Sequence[JAXArray]] = None,
    optimizations: Mapping = None,
):
    """Computes the implicit Hessian or chain rule depending on Dg, Hg,
    jvp_vec, using batched operations.

    Args:
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        z: the optimal embedding variable value array
        *params: the parameters p of the bilevel program
        nondiff_kw: nondifferentiable parameters to the implicit function
        Dg: gradient sensitivity vector (wrt z), for chain rule
        Hg: Hessian sensitivity vector (wrt z), for chain rule
        jvp_vec: right sensitivity vector(s) (wrt p) for Hessian-vector-product
        optimizations: optional optimizations
    Returns:
        Hessian/chain rule Hessian as specified by arguments
    """
    optimizations = {} if optimizations is None else copy(optimizations)
    nondiff_kw = {} if nondiff_kw is None else nondiff_kw

    blen, zlen = z.shape[0], prod(z.shape[1:])
    plen = [prod(param.shape[1:]) for param in params]

    jvp_vec = _ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec
    if jvp_vec is not None:
        assert Dg is not None

    if nondiff_kw is not None:
        k_fn_ = lambda z, *params: k_fn(z, *params, **nondiff_kw)
    else:
        k_fn_ = k_fn

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        _generate_default_Dzk_solve_fn(optimizations, k_fn_)
    #############################################################

    # compute 2nd implicit gradients
    if Dg is not None:
        assert Dg.size == zlen * blen
        assert Hg is None or Hg.size == blen * zlen ** 2

        Dg_ = Dg.reshape((blen, zlen, 1))
        Hg_ = Hg.reshape((blen, zlen, zlen)) if Hg is not None else Hg

        # compute the left hand vector in the VJP
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        v = -Dzk_solve_fn(z, *params, rhs=Dg_.reshape((blen, zlen, 1)), T=True)
        fn = lambda z, *params: jaxm.sum(
            v.reshape((blen, zlen)) * k_fn_(z, *params).reshape((blen, zlen))
        )

        if jvp_vec is not None:
            Dpz_jvp = _ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    nondiff_kw=nondiff_kw,
                    jvp_vec=jvp_vec,
                    optimizations=optimizations,
                )
            )
            Dpz_jvp = [Dpz_jvp.reshape(-1) for Dpz_jvp in Dpz_jvp]

            # compute the 2nd order derivatives consisting of 4 terms
            # term 1 ##############################
            dfn_params = jaxm.grad(
                lambda *params: fn(z, *params), argnums=range(len(params))
            )
            Dpp1 = _ensure_list(jaxm.jvp(dfn_params, params, tuple(jvp_vec))[1])
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
                    params,
                    tuple(jvp_vec),
                )[1]
            )
            Dpp3 = [
                _ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        nondiff_kw=nondiff_kw,
                        Dg=g_,
                        optimizations=optimizations,
                    )
                )[i].reshape((blen, plen))
                for (i, (g_, plen)) in enumerate(zip(g_, plen))
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
                    + bmv(Hg_, Dpz_jvp.reshape((blen, zlen)))
                    for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
                ]
            Dpp4 = [
                _ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        nondiff_kw=nondiff_kw,
                        optimizations=optimizations,
                    )
                )[i].reshape((blen, plen))
                for (i, (g_, plen)) in enumerate(zip(g_, plen))
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
            # compute the full first order 1st gradients
            Dpz = _ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    nondiff_kw=nondiff_kw,
                    optimizations=optimizations,
                )
            )
            Dpz = [
                Dpz.reshape((blen, zlen, plen))
                for (Dpz, plen) in zip(Dpz, plen)
            ]

            # compute the 2nd order derivatives consisting of 4 terms
            Dpp1 = BATCH_HESSIAN_DIAG(lambda *params: fn(z, *params))(*params)
            Dpp1 = [
                Dpp1.reshape((blen, plen, plen))
                for (Dpp1, plen) in zip(Dpp1, plen)
            ]

            temp = BATCH_JACOBIAN(
                lambda *params: jaxm.grad(fn)(z, *params),
                argnums=range(len(params)),
            )(*params)
            temp = [
                jaxm.t(temp.reshape((blen, zlen, plen)))
                for (temp, plen) in zip(temp, plen)
            ]
            Dpp2 = [
                (temp @ Dpz).reshape((blen, plen, plen))
                for (temp, Dpz, plen) in zip(temp, Dpz, plen)
            ]
            Dpp3 = [jaxm.t(Dpp2) for Dpp2 in Dpp2]
            Dzz = BATCH_HESSIAN(lambda z: fn(z, *params))(z).reshape(
                (blen, zlen, zlen)
            )
            if Hg is not None:
                Dpp4 = [jaxm.t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            else:
                Dpp4 = [jaxm.t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
            Dp = [Dg_.reshape((blen, 1, zlen)) @ Dpz for Dpz in Dpz]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [
                Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)
            ]
            Dpp_shaped = [
                Dpp.reshape((blen,) + 2 * param.shape[1:])
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
            nondiff_kw=nondiff_kw,
            full_output=True,
            optimizations=optimizations,
        )
        Dpz = _ensure_list(Dpz)
        Dpz = [
            Dpz.reshape((blen, zlen, plen)) for (Dpz, plen) in zip(Dpz, plen)
        ]

        # compute derivatives
        if optimizations.get("Dzzk", None) is None:
            Hk = BATCH_HESSIAN_DIAG(k_fn_)(z, *params)
            Dzzk, Dppk = Hk[0], Hk[1:]
            optimizations["Dzzk"] = Dzzk
        else:
            Dppk = BATCH_HESSIAN_DIAG(lambda *params: k_fn_(z, *params))(
                *params
            )
        Dppk = [
            Dppk.reshape((blen, zlen, plen, plen))
            for (Dppk, plen) in zip(Dppk, plen)
        ]
        Dzpk = BATCH_JACOBIAN(
            lambda *params: BATCH_JACOBIAN(k_fn_)(z, *params),
            argnums=range(len(params)),
        )(*params)
        Dzzk = Dzzk.reshape((blen, zlen, zlen, zlen))
        Dzpk = [
            Dzpk.reshape((blen, zlen, zlen, plen))
            for (Dzpk, plen) in zip(Dzpk, plen)
        ]
        Dpzk = [jaxm.t(Dzpk) for Dzpk in Dzpk]

        # solve the IFT equation
        lhs = [
            Dppk
            + Dpzk @ Dpz[:, None, ...]
            + jaxm.t(Dpz)[:, None, ...] @ Dzpk
            + (jaxm.t(Dpz)[:, None, ...] @ Dzzk) @ Dpz[:, None, ...]
            for (Dpz, Dzpk, Dpzk, Dppk) in zip(Dpz, Dzpk, Dpzk, Dppk)
        ]
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dppz = [
            -Dzk_solve_fn(
                z, *params, rhs=lhs.reshape((blen, zlen, plen * plen)), T=False
            ).reshape((blen, zlen, plen, plen))
            for (lhs, plen) in zip(lhs, plen)
        ]

        # return computed values
        Dpz_shaped = [
            Dpz.reshape((blen,) + z.shape[1:] + param.shape[1:])
            for (Dpz, param) in zip(Dpz, params)
        ]
        Dppz_shaped = [
            Dppz.reshape((blen,) + z.shape[1:] + 2 * param.shape[1:])
            for (Dppz, param) in zip(Dppz, params)
        ]
        return (
            (Dpz_shaped[0], Dppz_shaped[0])
            if len(params) == 1
            else (Dpz_shaped, Dppz_shaped)
        )


################################################################################


def generate_optimization_fns(
    loss_fn: Callable,
    opt_fn: Callable,
    k_fn: Callable,
    normalize_grad: bool = False,
    optimizations: Mapping = None,
    jit: bool = True,
    use_cache: bool = True,
    kw_in_key: bool = True,
):
    """Directly generates upper/outer bilevel program derivative functions.

    Args:
        loss_fn: loss_fn(z, *params), upper/outer level loss
        opt_fn: opt_fn(*params) = z, lower/inner argmin function
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        normalize_grad: whether to normalize the gradient by its norm
        jit: whether to apply just-in-time (jit) compilation to the functions
        cache_solutions: whether to cache the solution
    Returns:
        ``f_fn(*params), g_fn(*params), h_fn(*params)``
        parameters-only upper/outer level loss, gradient and Hessian.
    """
    sol_cache = dict()
    optimizations = dict() if optimizations is None else copy(optimizations)

    @fn_with_sol_cache(opt_fn, sol_cache, jit=jit, use_cache=use_cache, kw_in_key=kw_in_key)
    def f_fn(z, *params, **nondiff_kw):
        return loss_fn(z, *params, **nondiff_kw)

    @fn_with_sol_cache(opt_fn, sol_cache, jit=jit, use_cache=use_cache, kw_in_key=kw_in_key)
    def g_fn(z, *params, **nondiff_kw):
        g = JACOBIAN(loss_fn, argnums=range(len(params) + 1))(
            z, *params, **nondiff_kw
        )
        Dp = implicit_jacobian(
            k_fn,
            z,
            *params,
            nondiff_kw=nondiff_kw,
            Dg=g[0],
            optimizations=optimizations,
        )
        Dp = Dp if len(params) != 1 else [Dp]
        ret = [Dp + g for (Dp, g) in zip(Dp, g[1:])]
        if normalize_grad:
            ret = [(z / (jaxm.norm(z) + 1e-7)) for z in ret]
        return ret[0] if len(ret) == 1 else ret

    @fn_with_sol_cache(opt_fn, sol_cache, jit=jit, use_cache=use_cache, kw_in_key=kw_in_key)
    def h_fn(z, *params, **nondiff_kw):
        g = JACOBIAN(loss_fn, argnums=range(len(params) + 1))(
            z, *params, **nondiff_kw
        )

        if optimizations.get("Hz_fn", None) is None:
            loss_fn_ = lambda z, *params: loss_fn(z, *params, **nondiff_kw)
            optimizations["Hz_fn"] = BATCH_HESSIAN(loss_fn_)
        Hz_fn = optimizations["Hz_fn"]
        Hz = Hz_fn(z, *params)
        H = [Hz] + list(
            BATCH_HESSIAN_DIAG(
                lambda *params: loss_fn(z, *params, **nondiff_kw)
            )(*params)
        )

        _, Dpp = implicit_hessian(
            k_fn,
            z,
            *params,
            nondiff_kw=nondiff_kw,
            Dg=g[0],
            Hg=H[0],
            optimizations=optimizations,
        )
        Dpp = Dpp if len(params) != 1 else [Dpp]
        ret = [Dpp + H for (Dpp, H) in zip(Dpp, H[1:])]
        return ret[0] if len(ret) == 1 else ret

    return (f_fn, g_fn, h_fn)
