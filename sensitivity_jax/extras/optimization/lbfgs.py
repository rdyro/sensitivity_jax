# library imports and utils ########################################################################
from __future__ import annotations

from typing import Callable, Union, Optional, Dict, Any

import torch
import tqdm as tqdm_module
import tqdm.notebook  # pylint: disable=unused-import

from jfi import jaxm
import jaxopt

from ..extras_utils import x2t
from ...utils import t2j, TablePrinter

USE_TORCH = False

import jax
from jax import Array as JAXArray


# L-BFGS ###########################################################################################


def minimize_lbfgs(
    f_fn: Callable,
    g_fn: Callable,
    *args: JAXArray,
    verbose: bool = False,
    verbose_prefix: str = "",
    lr: float = 1e0,
    max_it: int = 100,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
    state: Optional[Dict[str, Any]] = None,
):
    """Minimize a loss function `f_fn` with L-BFGS with respect to `*args`.
    Taken from PyTorch.

    Args:
        f_fn: loss function
        g_fn: gradient of the loss function
        *args: arguments to be optimized
        verbose: whether to print output
        verbose_prefix: prefix to append to verbose output, e.g. indentation
        lr: learning rate, where 1.0 is unstable, use 1e-1 in most cases
        max_it: maximum number of iterates
        full_output: whether to output optimization history
        callback_fn: callback function of the form ``cb_fn(*args, **kw)``
        use_writer: whether to use tensorflow's Summary Writer (via PyTorch)
        use_tqdm: whether to use tqdm (to estimate total runtime)
    Returns:
        Optimized `args` or `(args, args_hist, state)` if `full_output` is `True`
    """
    state = dict() if state is None else state
    dtype = args[0].dtype
    device = args[0].device() if hasattr(args[0], "device") else "cpu"

    if isinstance(use_tqdm, bool):
        if use_tqdm:
            print_fn, rng_wrapper = tqdm_module.tqdm.write, tqdm_module.tqdm
        else:
            print_fn, rng_wrapper = print, lambda x: x
    else:
        print_fn, rng_wrapper = use_tqdm.write, use_tqdm

    assert len(args) > 0

    args = [x2t(arg).clone().detach() for arg in args] if USE_TORCH else args
    it, imprv = 0, float("inf")
    if USE_TORCH:
        opt = state.get("opt", torch.optim.LBFGS(args, lr=lr))
    else:
        assert len(args) == 1, "`jaxopt.LBFGS` only supports one argument"
        arg = args[0]

        @jax.jit
        def value_and_grad_fn(x):
            return lr * f_fn(x), lr * g_fn(x)

        opt = jaxopt.LBFGS(value_and_grad_fn, value_and_grad=True, jit=True)
        opt_state = opt.init_state(arg)
        opt_update_fn = jax.jit(opt.update)

    args_hist = [[arg.detach().clone() for arg in args]] if USE_TORCH else [arg]

    if callback_fn is not None:
        if USE_TORCH:
            callback_fn(*[t2j(arg) for arg in args], opt=opt)
        else:
            callback_fn(*[t2j(arg) for arg in args], opt=opt, opt_state=opt_state)

    if USE_TORCH:

        def closure():
            opt.zero_grad()
            args_ = [t2j(arg) for arg in args]
            args_ = [jaxm.to(t2j(arg), device=device, dtype=dtype) for arg in args]
            loss = torch.mean(x2t(f_fn(*args_)))
            gs = g_fn(*args_)
            gs = gs if isinstance(gs, list) or isinstance(gs, tuple) else [gs]
            gs = [x2t(g) for g in gs]
            for arg, g in zip(args, gs):
                arg.grad = torch.detach(g)
            return loss

    else:
        closure = None

    tp = TablePrinter(
        ["it", "imprv", "loss", "||g||_2"],
        ["%05d", "%9.4e", "%9.4e", "%9.4e"],
        prefix=verbose_prefix,
        use_writer=use_writer,
    )
    g_norm = float("inf")
    if verbose:
        print_fn(tp.make_header())
    try:
        for it in rng_wrapper(range(max_it)):
            if USE_TORCH:
                args_prev = [arg.detach().clone() for arg in args]
                loss = opt.step(closure)
                if full_output:
                    args_hist.append([arg.detach().clone() for arg in args])
                if callback_fn is not None:
                    callback_fn(*[t2j(arg) for arg in args])
                imprv = sum(
                    torch.norm(arg_prev - arg).detach() for (arg, arg_prev) in zip(args, args_prev)
                )
                if verbose or use_writer:
                    closure()
                    g_norm = sum(arg.grad.norm().detach() for arg in args if arg.grad is not None)
                imprv, loss = imprv.detach(), loss.detach()
            else:
                arg_prev = arg
                arg, opt_state = opt_update_fn(arg, opt_state)
                if full_output:
                    args_hist.append([arg])
                if callback_fn is not None:
                    callback_fn(arg, opt=opt, opt_state=opt_state)
                imprv = jaxm.norm(arg - arg_prev)
                loss = opt_state.value
                if verbose or use_writer:
                    g_norm = jaxm.norm(opt_state.grad)
            line = tp.make_values([it, imprv, loss, g_norm])
            if verbose:
                print_fn(line)
            if imprv < 1e-9:
                break
    except KeyboardInterrupt:
        pass
    if verbose:
        print_fn(tp.make_footer())
    if USE_TORCH:
        ret = [t2j(arg.detach()) for arg in args]
        ret = ret if len(args) > 1 else ret[0]
        args_hist = [[t2j(arg) for arg in z] for z in args_hist]
        args_hist = [z if len(args) > 1 else z[0] for z in args_hist]
        state = dict(state, opt=opt)
    else:
        state = dict(state, opt=opt, opt_state=opt_state)
        ret = arg
    return (ret, args_hist, state) if full_output else ret
