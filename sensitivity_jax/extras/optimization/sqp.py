# library imports and utils ########################################################################
from __future__ import annotations

import math
from typing import Callable, Union, Optional, Dict, Any

import numpy as np
import tqdm as tqdm_module
import tqdm.notebook  # pylint: disable=unused-import
from joblib import Parallel, delayed

from jfi import jaxm

from ...utils import TablePrinter
from .second_order_solvers import SQPSolver

from jax import Array
def minimize_sqp(
    f_fn: Callable,
    g_fn: Callable,
    h_fn: Callable,
    *args: Array,
    reg0: float = 1e-7,
    verbose: bool = False,
    verbose_prefix: str = "",
    max_it: int = 100,
    ls_pts_nb: int = 5,
    force_step: bool = False,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
    state: Optional[Dict[str, Any]] = None,
    parallel_ls: bool = False,
) -> Array | tuple[Array, list[Array], dict[str, Any]]:
    """
    Minimizes an unconstrained objective using Sequential Quadratic Programming (SQP).

    Args:
        f_fn (Callable): Objective function.
        g_fn (Callable): Gradient function.
        h_fn (Callable): Hessian function.
        *args (Array): Arguments.
        reg0 (float, optional): Regularization parameter. Defaults to 1e-7.
        verbose (bool, optional): If True, prints the optimization progress. Defaults to False.
        verbose_prefix (str, optional): Prefix to add to the printed progress. Defaults to "".
        max_it (int, optional): Maximum number of iterations. Defaults to 100.
        ls_pts_nb (int, optional): Number of points to use in the line search. Defaults to 5.
        force_step (bool, optional): Forces a step even if the line search fails. Defaults to False.
        full_output (bool, optional): If True, returns additional information. Defaults to False.
        callback_fn (Callable, optional): Ignored.
        use_writer (bool, optional): If True, ues a writer to print the progress. Defaults to False.
        use_tqdm (Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook], optional): If True, uses tqdm to print the progress. Defaults to False.
        state (Optional[Dict[str, Any]], optional): A dictionary containing the optimization state. Defaults to None.
        parallel_ls (bool, optional): If True, uses parallel line search. Defaults to False.

    Returns:
        Optimized parameters or a tuple with extra information if `full_output` is True.
    """

    state = state if state is not None else dict()
    arg = args[0]
    if "opt" not in state:
        opt = SQPSolver(
            #f_fn, g_fn=g_fn, h_fn=h_fn, linesearch="scan", maxls=ls_pts_nb, device=arg.device()
            f_fn, g_fn=g_fn, h_fn=h_fn, linesearch="scan", maxls=ls_pts_nb, force_step=force_step
        )
    else:
        opt = state["opt"]
    opt_state = opt.init_state(arg) if "opt_state" not in state else state["opt_state"]

    if isinstance(use_tqdm, bool):
        if use_tqdm:
            print_fn, rng_wrapper = tqdm_module.tqdm.write, tqdm_module.tqdm
        else:
            print_fn, rng_wrapper = print, lambda x, **kw: x
    else:
        print_fn, rng_wrapper = use_tqdm.write, use_tqdm
    tp = TablePrinter(
        ["it", "imprv", "loss", "reg_it", "bet", "||g_prev||_2"],
        ["%05d", "%9.4e", "%9.4e", "%02d", "%9.4e", "%9.4e"],
        prefix=verbose_prefix,
        use_writer=use_writer,
    )
    x_hist = [arg]
    if verbose:
        print_fn(tp.make_header())
        line = tp.make_values([0, 0, f_fn(arg), 0, 0.0, jaxm.norm(g_fn(arg))])
        print_fn(line)
    try:
        for it in rng_wrapper(
            range(max_it), disable=not (use_tqdm if isinstance(use_tqdm, bool) else True)
        ):
            new_arg, opt_state = opt.update(arg, opt_state)
            imprv = jaxm.norm(new_arg - arg)
            #loss = opt_state.best_loss
            loss = f_fn(new_arg)
            if verbose or use_writer:
                line = tp.make_values([it + 1, imprv, loss, 0, 0.0, jaxm.norm(g_fn(new_arg))])
                if verbose:
                    print_fn(line)
            arg = new_arg
            x_hist.append(arg)
            if imprv < 1e-9:
                break
    except (KeyboardInterrupt, InterruptedError):
        pass

    if verbose:
        print_fn(tp.make_footer())
    if full_output:
        return opt_state.best_params, x_hist + [opt_state.best_params], state
    else:
        return opt_state.best_params


# SQP (own) ########################################################################################


def _linesearch(
    f: Array,
    x: Array,
    d: Array,
    f_fn: Callable,
    g_fn: Callable | None = None,
    ls_pts_nb: int = 5,
    force_step: bool = False,
    parallel_ls: bool = False,
):
    """A linesearch routine

    Args:
        f (Array): Current best loss value
        x (Array): Current argument
        d (Array): Linesearch direction
        f_fn (Callable): Loss function
        g_fn (Callable | None, optional): Optional gradient function. Defaults to None.
        ls_pts_nb (int, optional): Number of linesearch points. Defaults to 5.
        force_step (bool, optional): Disallow non-zero step (even if worse)? Defaults to False.
        parallel_ls (bool, optional): Evaluate losses in parallel (threads). Defaults to False.

    Returns:
        _type_: _description_
    """
    opts = dict(dtype=x.dtype, device=x.device())
    if ls_pts_nb >= 2:
        bets = 10.0 ** jaxm.linspace(-1, 1, ls_pts_nb, **opts)
    else:
        bets = jaxm.to(jaxm.array([1.0]), **opts)

    if parallel_ls and ls_pts_nb > 2:
        xs = [x + bet * d for bet in bets]
        ys = Parallel(n_jobs=-1, backend="threading")(delayed(f_fn)(x) for x in xs)
        y = jaxm.stack([jaxm.atleast_1d(y) for y in ys], 1)
    else:
        y = jaxm.stack([f_fn(x + bet * d) for bet in bets])
    y = jaxm.where(jaxm.isnan(y), math.inf, y)

    if not force_step:
        zero_bet = bets.reshape(-1)[:1] * 0
        bets = jaxm.cat([zero_bet, bets], -1)
        y = jaxm.cat([jaxm.atleast_1d(f), y], -1)

    # idxs = jaxm.argmin(y, 1)
    # f_best = jaxm.to(jaxm.array([y[i, idx] for (i, idx) in enumerate(idxs)]), **opts)
    # bet = jaxm.to(jaxm.array([bets[idx] for idx in idxs]), **opts)
    idx = jaxm.argmin(y)
    f_best, bet = y[idx], bets[idx]

    d_norm = jaxm.norm(d.reshape((d.shape[0], -1)), axis=1)

    return bet, dict(d_norm=d_norm, f_best=f_best)


def _positive_factorization_cholesky(H, reg0):
    reg_it_max = 0
    reg, reg_it = reg0, 0
    H_reg, F = H, None
    opts = dict(dtype=H.dtype, device=H.device())
    while True:
        try:
            # H_reg = H + jaxm.diag(reg * jaxm.ones((H_reg.shape[-1],), **opts))
            H_reg = H + jaxm.diag(reg * jaxm.ones((H_reg.shape[-1],), **opts))
            F = jaxm.linalg.cholesky(H_reg)
            assert not jaxm.any(jaxm.isnan(F[0]))
            break
        except AssertionError:
            reg_it += 1
            reg *= 5e0
            if reg >= 0.99e7:
                raise RuntimeError("Numerical problems")
    reg_it_max = max(reg_it_max, reg_it)
    return F, (reg_it, reg)


def _positive_factorization_lobpcg(H, reg0):
    reg = jaxm.min(jaxm.linalg.eigvalsh(H.reshape((H.shape[-1], H.shape[-1]))).real)
    reg = reg.reshape(-1)[0]
    return _positive_factorization_cholesky(H, max(max(-2.0 * reg, 0.0), reg0))


####################################################################################################


def _minimize_sqp2(
    f_fn: Callable,
    g_fn: Callable,
    h_fn: Callable,
    *args: Array,
    reg0: float = 1e-7,
    verbose: bool = False,
    verbose_prefix: str = "",
    max_it: int = 100,
    ls_pts_nb: int = 5,
    force_step: bool = False,
    full_output: bool = False,
    callback_fn: Callable = None,
    use_writer: bool = False,
    use_tqdm: Union[bool, tqdm_module.std.tqdm, tqdm_module.notebook.tqdm_notebook] = False,
    state: Optional[Dict[str, Any]] = None,
    parallel_ls: bool = False,
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
        ls_pts_nb: number of linesearch points to consider per optimization step
        force_step: whether to take any non-zero optimization step even if worse
        full_output: whether to output optimization history
        callback_fn: callback function of the form ``cb_fn(*args, **kw)``
        use_writer: whether to use tensorflow's Summary Writer (via PyTorch)
        use_tqdm: whether to use tqdm (to estimate total runtime)
    Returns:
        Optimized ``args`` or ``(args, args_hist)`` if ``full_output`` is ``True``
    """
    state = dict() if state is None else state
    if isinstance(use_tqdm, bool):
        if use_tqdm:
            print_fn, rng_wrapper = tqdm_module.tqdm.write, tqdm_module.tqdm
        else:
            print_fn, rng_wrapper = print, lambda x, **kw: x
    else:
        print_fn, rng_wrapper = use_tqdm.write, use_tqdm

    if len(args) > 1:
        raise ValueError("SQP only supports single variable functions")
    x = args[0]
    x_shape = x.shape

    if callback_fn is not None:
        callback_fn(x)

    M, x_size = 1, x.size
    it, imprv = 0, float("inf")
    x_best, f_best = x, f_fn(x)
    f_hist, x_hist = [f_best], [x]

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

            # if jaxm.zeros(()).device().platform == "gpu":
            #    F, (reg_it_max, _) = _positive_factorization_cholesky(H, reg0)
            # else:
            #    F, (reg_it_max, _) = _positive_factorization_lobpcg(H, reg0)
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
                parallel_ls=parallel_ls,
            )

            x = x + jaxm.reshape(bet, (M,) + (1,) * len(x_shape[1:])) * d
            x_hist.append(x)
            imprv = jaxm.mean(bet * data["d_norm"])
            if callback_fn is not None:
                callback_fn(x)
            if data["f_best"] < f_best:
                x_best, f_best = x, data["f_best"]
            f_hist.append(data["f_best"])
            if verbose or use_writer:
                line = tp.make_values(
                    [it, imprv, jaxm.mean(data["f_best"]), reg_it_max, bet, jaxm.norm(g)]
                )
                if verbose:
                    print_fn(line)
            if imprv < 1e-9:
                break
    except (KeyboardInterrupt, InterruptedError):
        pass
    if verbose:
        print_fn(tp.make_footer())
    if full_output:
        return x_best, x_hist + [x_best], state
    else:
        return x_best
