sensitivity\_jax package
========================

Public API
==========

.. toctree::
   :maxdepth: 1
   :caption: Public API

.. currentmodule:: sensitivity_jax

Sensitivity Analysis (:code:`sensitivity`)
------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  sensitivity.implicit_jacobian
  sensitivity.implicit_hessian
  sensitivity.generate_optimization_fns

Differentiation (:code:`differentiation`)
-----------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  differentiation.JACOBIAN
  differentiation.HESSIAN
  differentiation.HESSIAN_DIAG

Jax Friendly Interface (:code:`jax_friendly_interface`)
-------------------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  jax_friendly_interface.init
  jax_friendly_interface.manual_seed

Extras: Optimization (:code:`extras.optimization`)
--------------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  extras.optimization.minimize_agd
  extras.optimization.minimize_lbfgs
  extras.optimization.minimize_sqp

Extras: Neural Network Tools (:code:`extras.nn_tools`)
----------------------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  extras.nn_tools.nn_all_params
  extras.nn_tools.nn_forward_gen


..  Subpackages
    -----------

    .. toctree::
       :maxdepth: 4

       sensitivity_jax.extras

    Submodules
    ----------

    .. toctree::
       :maxdepth: 4

       sensitivity_jax.differentiation
       sensitivity_jax.jax_friendly_interface
       sensitivity_jax.sensitivity
       sensitivity_jax.specialized_matrix_inverse
       sensitivity_jax.utils

    Module contents
    ---------------

    .. automodule:: sensitivity_jax
       :members:
       :undoc-members:
       :show-inheritance:
