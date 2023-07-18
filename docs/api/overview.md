# `sensitivity`

|                                                                                                      name                                                                                                       |                                  summary                                  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
|            [generate_optimization_fns(loss_fn, opt_fn, k_fn, normalize_grad, optimizations, jit, custom_arg_serializer)](/sensitivity_jax/api/sensitivity_jax/sensitivity/generate_optimization_fns)            |   Directly generates upper/outer bilevel program derivative functions.    |
| [generate_optimization_with_state_fns(loss_fn, opt_fn, k_fn, normalize_grad, optimizations, jit, custom_arg_serializer)](/sensitivity_jax/api/sensitivity_jax/sensitivity/generate_optimization_with_state_fns) |   Directly generates upper/outer bilevel program derivative functions.    |
|                               [implicit_hessian(k_fn, z, params, nondiff_kw, Dg, Hg, jvp_vec, optimizations)](/sensitivity_jax/api/sensitivity_jax/sensitivity/implicit_hessian)                                | Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec. |
|               [implicit_jacobian(k_fn, z, params, nondiff_kw, Dg, jvp_vec, matrix_free_inverse, full_output, optimizations)](/sensitivity_jax/api/sensitivity_jax/sensitivity/implicit_jacobian)                |  Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec.   |



# `batch_sensitivity`

|                                                                                              name                                                                                              |                                               summary                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [generate_optimization_fns(loss_fn, opt_fn, k_fn, normalize_grad, optimizations, jit, use_cache, kw_in_key)](/sensitivity_jax/api/sensitivity_jax/batch_sensitivity/generate_optimization_fns) |                Directly generates upper/outer bilevel program derivative functions.                 |
|                    [implicit_hessian(k_fn, z, params, nondiff_kw, Dg, Hg, jvp_vec, optimizations)](/sensitivity_jax/api/sensitivity_jax/batch_sensitivity/implicit_hessian)                    | Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec, using batched operations. |
|                   [implicit_hessian2(k_fn, z, params, nondiff_kw, Dg, Hg, jvp_vec, optimizations)](/sensitivity_jax/api/sensitivity_jax/batch_sensitivity/implicit_hessian2)                   |        Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec, using vmap.        |
|    [implicit_jacobian(k_fn, z, params, nondiff_kw, Dg, jvp_vec, matrix_free_inverse, full_output, optimizations)](/sensitivity_jax/api/sensitivity_jax/batch_sensitivity/implicit_jacobian)    |  Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec, using batched operations.   |
|   [implicit_jacobian2(k_fn, z, params, nondiff_kw, Dg, jvp_vec, matrix_free_inverse, full_output, optimizations)](/sensitivity_jax/api/sensitivity_jax/batch_sensitivity/implicit_jacobian2)   |         Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec, using vmap.          |



# `extras.optimization.agd`

|                                                                                                          name                                                                                                          |                                                      summary                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [minimize_agd(f_fn, g_fn, args, verbose, verbose_prefix, max_it, ai, af, full_output, callback_fn, use_writer, use_tqdm, state, optimizer)](/sensitivity_jax/api/sensitivity_jax/extras/optimization/agd/minimize_agd) | Minimize a loss function ``f_fn`` with Accelerated Gradient Descent (AGD) with respect to ``*args``. Uses PyTorch. |


# `extras.optimization.lbfgs`

|                                                                                                     name                                                                                                      |                                         summary                                          |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [minimize_lbfgs(f_fn, g_fn, args, verbose, verbose_prefix, lr, max_it, full_output, callback_fn, use_writer, use_tqdm, state)](/sensitivity_jax/api/sensitivity_jax/extras/optimization/lbfgs/minimize_lbfgs) | Minimize a loss function `f_fn` with L-BFGS with respect to `*args`. Taken from PyTorch. |


# `extras.optimization.sqp`

|                                                                                                                        name                                                                                                                         |                                       summary                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [minimize_sqp(f_fn, g_fn, h_fn, args, reg0, verbose, verbose_prefix, max_it, ls_pts_nb, force_step, full_output, callback_fn, use_writer, use_tqdm, state, parallel_ls)](/sensitivity_jax/api/sensitivity_jax/extras/optimization/sqp/minimize_sqp) |  Minimizes an unconstrained objective using Sequential Quadratic Programming (SQP). |



# `utils`

|                                                                                  name                                                                                  |                                           summary                                           |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [fn_with_sol_and_state_cache(fwd_fn, cache, jit, use_cache, kw_in_key, custom_arg_serializer)](/sensitivity_jax/api/sensitivity_jax/utils/fn_with_sol_and_state_cache) | Wraps a function in a version where computation of the first argument via fwd_fn is cached. |
|           [fn_with_sol_cache(fwd_fn, cache, jit, use_cache, kw_in_key, custom_arg_serializer)](/sensitivity_jax/api/sensitivity_jax/utils/fn_with_sol_cache)           | Wraps a function in a version where computation of the first argument via fwd_fn is cached. |


# `differentiation`

|                                               name                                                |                                 summary                                 |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
|  [BATCH_HESSIAN(fn, config)](/sensitivity_jax/api/sensitivity_jax/differentiation/BATCH_HESSIAN)  | Computes the Hessian, assuming the first in/out dimension is the batch. |
| [BATCH_JACOBIAN(fn, config)](/sensitivity_jax/api/sensitivity_jax/differentiation/BATCH_JACOBIAN) | Computes the Hessian, assuming the first in/out dimension is the batch. |
|   [HESSIAN_DIAG(fn, config)](/sensitivity_jax/api/sensitivity_jax/differentiation/HESSIAN_DIAG)   |   Generates a function which computes per-argument partial Hessians.    |
