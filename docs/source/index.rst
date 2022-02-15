.. SecondOrderSensitivity-JAX documentation master file, created by
   sphinx-quickstart on Tue Feb 15 11:49:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SecondOrderSensitivity-JAX Documentation
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Quick Intro Tutorial
====================

.. toctree::
   :maxdepth: 1
   :caption: Quick Intro Tutorial

This package is designed to streamline...

Public API
==========

.. toctree::
   :maxdepth: 1
   :caption: Public API

.. currentmodule:: sos_jax

Sensitivity Analysis (:code:`sensitivity`)
------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  sensitivity.implicit_jacobian
  sensitivity.implicit_hessian
  sensitivity.generate_fns

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
