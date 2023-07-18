# library imports and utils ########################################################################
from __future__ import annotations

from typing import Callable, Union, Optional, Dict, Any

import tqdm as tqdm_module
import tqdm.notebook  # pylint: disable=unused-import

from jfi import jaxm
import optax

from ..extras_utils import x2t
from ...utils import t2j, TablePrinter

import jax
from jax import Array as JAXArray

apply_updates = jax.jit(optax.apply_updates)

USE_TORCH = True

if USE_TORCH:
    import torch

    OPT_MAP = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    DEFAULT_OPTIMIZER = OPT_MAP["adam"]
else:
    OPT_MAP = {
        "adam": optax.adam,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
        "adagrad": optax.adagrad,
    }
    DEFAULT_OPTIMIZER = OPT_MAP["adam"]

# Accelerated Gradient Descent #####################################################################


def minimize_agd(
    f_fn: Callable,
    g_fn: Callable,
    *args: JAXArray,
    verbose: bool = False,
    verbose_prefix: str = "",
    max_it: int = 10**3,
    ai: float = 1e-1,
    af: float = 1e-2,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
    state: Optional[Dict[str, Any]] = None,
    optimizer: str = "Adam",
):
    """Minimize a loss function ``f_fn`` with Accelerated Gradient Descent (AGD)
    with respect to ``*args``. Uses PyTorch.

    Args:
        f_fn: loss function
        g_fn: gradient of the loss function
        *args: arguments to be optimized
        verbose: whether to print output
        verbose_prefix: prefix to append to verbose output, e.g. indentation
        max_it: maximum number of iterates
        ai: initial gradient step length (exponential schedule)
        af: final gradient step length (exponential schedule)
        full_output: whether to output optimization history
        callback_fn: callback function of the form ``cb_fn(*args, **kw)``
        use_writer: whether to use tensorflow's Summary Writer (via PyTorch)
        use_tqdm: whether to use tqdm (to estimate total runtime)
    Returns:
        Optimized ``args`` or ``(args, args_hist)`` if ``full_output`` is ``True``
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

    args = [x2t(arg).clone().detach() for arg in args] if USE_TORCH else list(args)
    imprv = float("inf")
    gam = (af / ai) ** (1.0 / max_it)
    if USE_TORCH:
        opt = state.get("opt", OPT_MAP.get(optimizer.lower(), DEFAULT_OPTIMIZER)(args, lr=ai))
    else:
        opt = state.get("opt", OPT_MAP.get(optimizer.lower(), DEFAULT_OPTIMIZER)(ai))
        opt_state = state.get("opt_state", opt.init(args))
        opt_update_fn = jax.jit(opt.update)
    tp = TablePrinter(
        ["it", "imprv", "loss", "||g||_2"],
        ["%05d", "%9.4e", "%9.4e", "%9.4e"],
        prefix=verbose_prefix,
        use_writer=use_writer,
    )
    args_hist = [[arg.detach().clone() for arg in args]] if USE_TORCH else [[arg for arg in args]]

    if callback_fn is not None:
        if USE_TORCH:
            callback_fn(*[t2j(arg) for arg in args], opt=opt)
        else:
            callback_fn(*args, opt=opt, opt_state=opt_state)

    if verbose:
        print_fn(tp.make_header())
    try:
        for it in rng_wrapper(range(max_it)):
            if USE_TORCH:
                args_prev = [arg.clone().detach() for arg in args]
                opt.zero_grad()
                args_ = [jaxm.to(t2j(arg), device=device, dtype=dtype) for arg in args]
                loss = torch.mean(x2t(f_fn(*args_)))
                gs = g_fn(*args_)
                gs = gs if isinstance(gs, list) or isinstance(gs, tuple) else [gs]
                gs = [x2t(g) for g in gs]
                for arg, g in zip(args, gs):
                    arg.grad = torch.detach(g)
                g_norm = sum(
                    torch.norm(arg.grad) for arg in args if arg.grad is not None
                ).detach() / len(args)
                opt.step()
                args_hist.append([arg.detach().clone() for arg in args])
                if callback_fn is not None:
                    callback_fn(*[t2j(arg) for arg in args], opt=opt)
                imprv = sum(torch.norm(arg_prev - arg) for (arg, arg_prev) in zip(args, args_prev))
                imprv, loss = imprv.detach(), loss.detach()
            else:
                args_prev = args
                loss = jaxm.mean(f_fn(*args))
                gs = g_fn(*args)
                gs = gs if isinstance(gs, list) or isinstance(gs, tuple) else [gs]
                g_norm = sum(jaxm.norm(g) for g in gs) / len(gs)
                updates, opt_state = opt_update_fn([ai * gam**it * g for g in gs], opt_state)
                args = apply_updates(args, updates)
                args_hist.append(args)
                if callback_fn is not None:
                    callback_fn(*[t2j(arg) for arg in args], opt=opt, opt_state=opt_state)
                imprv = sum(jaxm.norm(arg_prev - arg) for (arg, arg_prev) in zip(args, args_prev))
            if verbose or use_writer:
                line = tp.make_values([it, imprv, loss, g_norm])
                if verbose:
                    print_fn(line)
            if USE_TORCH:
                for pgroup in opt.param_groups:
                    pgroup["lr"] *= gam
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
        ret = args if len(args) > 1 else args[0]
        args_hist = [z if len(args) > 1 else z[0] for z in args_hist]
        state = dict(state, opt=opt, opt_state=opt_state)

    return (ret, args_hist, state) if full_output else ret
