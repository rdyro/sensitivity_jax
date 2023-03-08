##^# library imports and utils #################################################
import math, os, pdb, sys, time
from typing import Callable, Union
import signal

import torch, numpy as np
import tqdm as tqdm_module
import tqdm.notebook
import optax

from ..jax_friendly_interface import init

jaxm = init()

from .extras_utils import x2t
from ..utils import t2j, TablePrinter

from jax import Array as JAXArray

##$#############################################################################
##^# Accelerated Gradient Descent ##############################################
def minimize_agd(
    f_fn: Callable,
    g_fn: Callable,
    *args: JAXArray,
    verbose: bool = False,
    verbose_prefix: str = "",
    max_it: int = 10**3,
    ai: float = 1e-1,
    af: float = 1e-2,
    batched: bool = False,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
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
        batched: whether to optimize a batch of arguments, with batch of losses
        full_output: whether to output optimization history
        callback_fn: callback function of the form ``cb_fn(*args, **kw)``
        use_writer: whether to use tensorflow's Summary Writer (via PyTorch)
        use_tqdm: whether to use tqdm (to estimate total runtime)
    Returns:
        Optimized ``args`` or ``(args, args_hist)`` if ``full_output`` is ``True``
    """

    if isinstance(use_tqdm, bool):
        if use_tqdm:
            print_fn, rng_wrapper = tqdm_module.tqdm.write, tqdm_module.tqdm
        else:
            print_fn, rng_wrapper = print, lambda x: x
    else:
        print_fn, rng_wrapper = use_tqdm.write, use_tqdm

    assert len(args) > 0
    assert g_fn is not None

    args = [x2t(arg) for arg in args]
    args = [arg.clone().detach() for arg in args]
    imprv = float("inf")
    gam = (af / ai) ** (1.0 / max_it)
    opt = torch.optim.Adam(args, lr=ai)
    tp = TablePrinter(
        ["it", "imprv", "loss", "||g||_2"],
        ["%05d", "%9.4e", "%9.4e", "%9.4e"],
        prefix=verbose_prefix,
        use_writer=use_writer,
    )
    args_hist = [[arg.detach().clone() for arg in args]]

    cb_kw = dict(opt=opt)
    if callback_fn is not None:
        callback_fn(*[t2j(arg) for arg in args], **cb_kw)

    if verbose:
        print_fn(tp.make_header())
    try:
        for it in rng_wrapper(range(max_it)):
            args_prev = [arg.clone().detach() for arg in args]
            opt.zero_grad()
            args_ = [t2j(arg) for arg in args]
            l = torch.mean(x2t(f_fn(*args_)))
            gs = g_fn(*args_)
            gs = gs if isinstance(gs, list) or isinstance(gs, tuple) else [gs]
            gs = [x2t(g) for g in gs]
            for (arg, g) in zip(args, gs):
                arg.grad = torch.detach(g)
            g_norm = sum(torch.norm(arg.grad) for arg in args if arg.grad is not None).detach() / len(
                args
            )
            opt.step()
            args_hist.append([arg.detach().clone() for arg in args])
            if callback_fn is not None:
                callback_fn(*[t2j(arg) for arg in args], **cb_kw)
            if batched:
                imprv = sum(
                    torch.mean(torch.norm(arg_prev - arg, dim=tuple(range(-(arg.ndim - 1), 0))))
                    for (arg, arg_prev) in zip(args, args_prev)
                )
            else:
                imprv = sum(torch.norm(arg_prev - arg) for (arg, arg_prev) in zip(args, args_prev))
            if verbose or use_writer:
                line = tp.make_values([it, imprv.detach(), l.detach(), g_norm])
                if verbose:
                    print_fn(line)
            for pgroup in opt.param_groups:
                pgroup["lr"] *= gam
    except KeyboardInterrupt:
        pass
    if verbose:
        print_fn(tp.make_footer())
    ret = [t2j(arg.detach()) for arg in args]
    ret = ret if len(args) > 1 else ret[0]
    args_hist = [[t2j(arg) for arg in z] for z in args_hist]
    args_hist = [z if len(args) > 1 else z[0] for z in args_hist]
    if full_output:
        return ret, args_hist
    else:
        return ret


##$#############################################################################
##^# L-BFGS ####################################################################
def minimize_lbfgs(
    f_fn: Callable,
    g_fn: Callable,
    *args: JAXArray,
    verbose: bool = False,
    verbose_prefix: str = "",
    lr: float = 1e0,
    max_it: int = 100,
    batched: bool = False,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
):
    """Minimize a loss function ``f_fn`` with L-BFGS with respect to ``*args``.
    Taken from PyTorch.

    Args:
        f_fn: loss function
        g_fn: gradient of the loss function
        *args: arguments to be optimized
        verbose: whether to print output
        verbose_prefix: prefix to append to verbose output, e.g. indentation
        lr: learning rate, where 1.0 is unstable, use 1e-1 in most cases
        max_it: maximum number of iterates
        batched: whether to optimize a batch of arguments, with batch of losses
        full_output: whether to output optimization history
        callback_fn: callback function of the form ``cb_fn(*args, **kw)``
        use_writer: whether to use tensorflow's Summary Writer (via PyTorch)
        use_tqdm: whether to use tqdm (to estimate total runtime)
    Returns:
        Optimized ``args`` or ``(args, args_hist)`` if ``full_output`` is ``True``
    """
    if isinstance(use_tqdm, bool):
        if use_tqdm:
            print_fn, rng_wrapper = tqdm_module.tqdm.write, tqdm_module.tqdm
        else:
            print_fn, rng_wrapper = print, lambda x: x
    else:
        print_fn, rng_wrapper = use_tqdm.write, use_tqdm

    assert len(args) > 0
    assert g_fn is not None

    args = [x2t(arg) for arg in args]
    args = [arg.clone().detach() for arg in args]

    if callback_fn is not None:
        callback_fn(*[t2j(arg) for arg in args])

    args = [x2t(arg) for arg in args]
    args = [arg.detach().clone() for arg in args]
    it, imprv = 0, float("inf")
    opt = torch.optim.LBFGS(args, lr=lr)
    args_hist = [[arg.detach().clone() for arg in args]]

    def closure():
        opt.zero_grad()
        args_ = [t2j(arg) for arg in args]
        l = torch.mean(x2t(f_fn(*args_)))
        gs = g_fn(*args_)
        gs = gs if isinstance(gs, list) or isinstance(gs, tuple) else [gs]
        gs = [x2t(g) for g in gs]
        for (arg, g) in zip(args, gs):
            arg.grad = torch.detach(g)
        return l

    tp = TablePrinter(
        ["it", "imprv", "loss", "||g||_2"],
        ["%05d", "%9.4e", "%9.4e", "%9.4e"],
        prefix=verbose_prefix,
        use_writer=use_writer,
    )
    if verbose:
        print_fn(tp.make_header())
    try:
        for it in rng_wrapper(range(max_it)):
            args_prev = [arg.detach().clone() for arg in args]
            l = opt.step(closure)
            if full_output:
                args_hist.append([arg.detach().clone() for arg in args])
            if callback_fn is not None:
                callback_fn(*[t2j(arg) for arg in args])
            if batched:
                imprv = sum(
                    torch.mean(torch.norm(arg_prev - arg, dim=tuple(range(-(arg.ndim - 1), 0))))
                    for (arg, arg_prev) in zip(args, args_prev)
                )
            else:
                imprv = sum(
                    torch.norm(arg_prev - arg).detach() for (arg, arg_prev) in zip(args, args_prev)
                )
            if verbose or use_writer:
                closure()
                g_norm = sum(arg.grad.norm().detach() for arg in args if arg.grad is not None)
                line = tp.make_values([it, imprv.detach(), l.detach(), g_norm])
                if verbose:
                    print_fn(line)
            if imprv < 1e-9:
                break
    except KeyboardInterrupt:
        pass
    if verbose:
        print_fn(tp.make_footer())
    ret = [t2j(arg.detach()) for arg in args]
    ret = ret if len(args) > 1 else ret[0]
    args_hist = [[t2j(arg) for arg in z] for z in args_hist]
    args_hist = [z if len(args) > 1 else z[0] for z in args_hist]
    if full_output:
        return ret, args_hist
    else:
        return ret


##$#############################################################################
##^# SQP (own) #################################################################
def _linesearch(
    f,
    x,
    d,
    f_fn,
    g_fn=None,
    ls_pts_nb=5,
    force_step=False,
):
    if ls_pts_nb >= 2:
        bets = 10.0 ** jaxm.linspace(-1, 1, ls_pts_nb)
    else:
        bets = jaxm.array([1.0])
    y = jaxm.stack(
        [jaxm.atleast_1d(f_fn(x + bet * d)) for bet in bets],
        1,
    )
    y = jaxm.where(jaxm.isnan(y), math.inf, y)

    if not force_step:
        bets = jaxm.cat([jaxm.zeros((1,)), bets], -1)
        y = jaxm.cat([jaxm.atleast_1d(f)[..., None], y], -1)

    idxs = jaxm.argmin(y, 1)
    f_best = jaxm.array([y[i, idx] for (i, idx) in enumerate(idxs)])
    bet = jaxm.array([bets[idx] for idx in idxs])

    d_norm = jaxm.norm(d.reshape((d.shape[0], -1)), axis=1)

    return bet, dict(d_norm=d_norm, f_best=f_best)


def _positive_factorization_cholesky(H, reg0):
    reg_it_max = 0
    reg, reg_it = reg0, 0
    H_reg, F = H, None
    while True:
        try:
            H_reg = H + jaxm.diag(reg * jaxm.ones((H_reg.shape[-1],)))
            F = jaxm.linalg.cholesky(H_reg)
            assert not jaxm.any(jaxm.isnan(F[0]))
            break
        except AssertionError as e:
            reg_it += 1
            reg *= 5e0
            if reg >= 0.99e7:
                raise RuntimeError("Numerical problems")
    reg_it_max = max(reg_it_max, reg_it)
    return F, (reg_it, reg)


def _positive_factorization_lobpcg(H, reg0):
    reg = jaxm.min(jaxm.linalg.eigvals(H.reshape((H.shape[-1], H.shape[-1]))).real)
    reg = reg.reshape(-1)[0]
    return _positive_factorization_cholesky(H, max(max(-2.0 * reg, 0.0), reg0))


def minimize_sqp(
    f_fn: Callable,
    g_fn: Callable,
    h_fn: Callable,
    *args: JAXArray,
    reg0: float = 1e-7,
    verbose: bool = False,
    verbose_prefix: str = "",
    max_it: int = 100,
    ls_pts_nb: int = 5,
    force_step: bool = False,
    batched: bool = False,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
):
    """Minimize a loss function ``f_fn`` with Unconstrained Sequential Quadratic
    Programming (SQP) with respect to a single ``arg``.

    Args:
        f_fn: loss function
        g_fn: gradient of the loss function
        h_fn: Hessian of the loss function
        *args: arguments to be optimized
        reg0: Hessian regularization – optimization step regularization
        verbose: whether to print output
        verbose_prefix: prefix to append to verbose output, e.g. indentation
        max_it: maximum number of iterates
        batched: whether to optimize a batch of arguments, with batch of losses
        ls_pts_nb: number of linesearch points to consider per optimization step
        force_step: whether to take any non-zero optimization step even if worse
        full_output: whether to output optimization history
        callback_fn: callback function of the form ``cb_fn(*args, **kw)``
        use_writer: whether to use tensorflow's Summary Writer (via PyTorch)
        use_tqdm: whether to use tqdm (to estimate total runtime)
    Returns:
        Optimized ``args`` or ``(args, args_hist)`` if ``full_output`` is ``True``
    """
    if isinstance(use_tqdm, bool):
        if use_tqdm:
            print_fn, rng_wrapper = tqdm_module.tqdm.write, tqdm_module.tqdm
        else:
            print_fn, rng_wrapper = print, lambda x, **kw: x
    else:
        print_fn, rng_wrapper = use_tqdm.write, use_tqdm

    if len(args) > 1:
        raise ValueError("SQP only only supports single variable functions")
    x = args[0]
    x_shape = x.shape

    if callback_fn is not None:
        callback_fn(x)

    if batched:
        M, x_size = x_shape[0], np.prod(x_shape[1:])
    else:
        M, x_size = 1, x.size
    it, imprv = 0, float("inf")
    x_best, f_best = x, jaxm.atleast_1d(f_fn(x))
    f_hist, x_hist = [f_best], [x]

    t__ = time.time()
    tp = TablePrinter(
        ["it", "imprv", "loss", "reg_it", "bet", "||g_prev||_2"],
        ["%05d", "%9.4e", "%9.4e", "%02d", "%9.4e", "%9.4e"],
        prefix=verbose_prefix,
        use_writer=use_writer,
    )
    if verbose:
        print_fn(tp.make_header())
    try:
        for it in rng_wrapper(
            range(max_it), disable=not (use_tqdm if isinstance(use_tqdm, bool) else True)
        ):
            g = g_fn(x).reshape((M, x_size))
            H = h_fn(x).reshape((M, x_size, x_size))
            if jaxm.any(jaxm.isnan(g)):
                raise RuntimeError("Gradient is NaN")
            if jaxm.any(jaxm.isnan(H)):
                raise RuntimeError("Hessian is NaN")

            if jaxm.zeros(()).device().platform == "gpu":
                F, (reg_it_max, _) = _positive_factorization_cholesky(H, reg0)
            else:
                F, (reg_it_max, _) = _positive_factorization_lobpcg(H, reg0)

            d = jaxm.linalg.cholesky_solve(F, -g[..., None])[..., 0].reshape(x_shape)
            f = f_hist[-1]
            bet, data = _linesearch(
                f,
                x,
                d,
                f_fn,
                g_fn,
                ls_pts_nb=ls_pts_nb,
                force_step=force_step,
            )

            x = x + jaxm.reshape(bet, (M,) + (1,) * len(x_shape[1:])) * d
            x_hist.append(x)
            imprv = jaxm.mean(bet * data["d_norm"])
            if callback_fn is not None:
                callback_fn(x)
            if batched:
                x_bests = [None for _ in range(M)]
                f_bests = [None for _ in range(M)]
                for i in range(M):
                    if data["f_best"][i] < f_best[i]:
                        x_bests[i], f_bests[i] = x[i, ...], data["f_best"][i]
                    else:
                        x_bests[i], f_bests[i] = x_best[i, ...], f_best[i]
                x_best, f_best = jaxm.stack(x_bests), jaxm.stack(f_bests)
            else:
                if data["f_best"][0] < f_best[0]:
                    x_best, f_best = x, data["f_best"]
            f_hist.append(data["f_best"])
            if verbose or use_writer:
                line = tp.make_values(
                    [
                        it,
                        imprv,
                        jaxm.mean(data["f_best"]),
                        reg_it_max,
                        bet[0],
                        jaxm.norm(g),
                    ]
                )
                if verbose:
                    print_fn(line)
            if imprv < 1e-9:
                break
    except KeyboardInterrupt:
        pass
    if verbose:
        print_fn(tp.make_footer())
    if full_output:
        return x_best, x_hist + [x_best]
    else:
        return x_best


##$#############################################################################
