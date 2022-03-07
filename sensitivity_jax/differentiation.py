from functools import reduce
from operator import mul
import pdb

from .jax_friendly_interface import init

jaxm = init()

JACOBIAN = jaxm.jacobian
JACOBIAN.__doc__ = """Equivalent to jax.jacobian."""
HESSIAN = jaxm.hessian
HESSIAN.__doc__ = """Equivalent to jax.hessian."""

prod = lambda x: reduce(mul, x, 1)


def HESSIAN_DIAG(fn, **config):
    """Generates a function which computes per-argument partial Hessians."""

    def h_fn(*args, **kw):
        args = (args,) if not isinstance(args, (tuple, list)) else tuple(args)
        ret = [
            jaxm.hessian(
                lambda arg: fn(*args[:i], arg, *args[i + 1 :], **kw), **config
            )(arg)
            for (i, arg) in enumerate(args)
        ]
        return ret

    return h_fn


# experimental #################################################################


def BATCH_JACOBIAN(fn, **config):
    """Computes the Hessian, assuming the first in/out dimension is the batch."""

    def batch_jac(*args, **kw):
        ret = jaxm.jacobian(
            lambda *args, **kw: jaxm.sum(jaxm.atleast_1d(fn(*args, **kw)), 0),
            **config
        )(*args, **kw)
        argnums = config.get("argnums", 0)
        argnums = list(argnums) if hasattr(argnums, "__iter__") else argnums

        Js, ret_struct = jaxm.jax.tree_flatten(ret)
        argnums, argnums_struct = jaxm.jax.tree_flatten(argnums)

        out_shapes = [
            J.shape[: -len(args[argnum].shape)]
            for (J, argnum) in zip(Js, argnums)
        ]
        Js = [
            J.reshape((prod(out_shape),) + args[argnum].shape)
            .swapaxes(0, 1)
            .reshape(
                (args[argnum].shape[0],) + out_shape + args[argnum].shape[1:]
            )
            for (J, out_shape, argnum) in zip(Js, out_shapes, argnums)
        ]
        ret = jaxm.jax.tree_unflatten(ret_struct, Js)
        return ret

    return batch_jac


def BATCH_HESSIAN(fn, **config):
    """Computes the Hessian, assuming the first in/out dimension is the batch."""

    def batch_hess(*args, **kw):
        return BATCH_JACOBIAN(BATCH_JACOBIAN(fn, **config), **config)(
            *args, **kw
        )

    return batch_hess


# def BATCH_HESSIAN(fn, **config):
#   """Computes the Hessian, assuming the first in/out dimension is the batch."""
#   dfn = lambda *args, **kw: jaxm.sum(
#       jaxm.jacobian(fn, **config)(*args, **kw), 0
#   ).reshape(-1)
#
#   def batch_hess(*args, **kw):
#       argnum = config.get("argnums", 0)
#       assert isinstance(argnum, int)
#
#       ret = jaxm.jacobian(dfn, **config)(*args, **kw).swapaxes(0, 1)
#       ret = ret.reshape((args[argnum].shape[0],) + 2 * args[argnum].shape[1:])
#       return ret
#
#   return batch_hess


# def BATCH_HESSIAN(fn, **config):
#    """Computes the Hessian, assuming the first in/out dimension is the batch."""
#    fn_ = lambda *args, **kw: jaxm.sum(fn(*args, **kw), 0)
#
#    def batch_hess(*args, **kw):
#        return jaxm.hessian(fn_, **config)(*args, **kw)
#
#    return batch_hess


def BATCH_HESSIAN_DIAG(fn, **config):
    def batch_hess_diag(*args, **kw):
        args, arg_struct = jaxm.jax.tree_flatten(args)
        ret = [
            BATCH_HESSIAN(
                lambda arg: fn(*args[:i], arg, *args[i + 1 :], **kw), **config
            )(arg)
            for (i, arg) in enumerate(args)
        ]
        return jaxm.jax.tree_unflatten(arg_struct, ret)

    return batch_hess_diag
