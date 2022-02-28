from .jax_friendly_interface import init

jaxm = init()

JACOBIAN = jaxm.jacobian
JACOBIAN.__doc__ = """Equivalent to jax.jacobian."""
HESSIAN = jaxm.hessian
HESSIAN.__doc__ = """Equivalent to jax.hessian."""


def HESSIAN_DIAG(fn):
    """Generates a function which computes per-argument partial Hessians."""

    def h_fn(*args, **kwargs):
        args = (args,) if not isinstance(args, (tuple, list)) else tuple(args)
        ret = [
            jaxm.hessian(
                lambda arg: fn(*args[:i], arg, *args[i + 1 :], **kwargs)
            )(arg)
            for (i, arg) in enumerate(args)
        ]
        return ret

    return h_fn
