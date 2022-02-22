import os
from setuptools import setup

# borrowed from https://pythonhosted.org/an_example_pypi_project/setuptools.html


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except FileNotFoundError:
        return ""


setup(
    name="sensitivity_jax",
    version="0.3.0",
    author="Robert Dyro",
    description=(
        "Optimization Sensitivity Analysis for Bilevel Programming for JAX"
    ),
    license="MIT",
    packages=["sensitivity_jax", "sensitivity_jax.extras"],
    #install_requires=["jax", "numpy", "tqdm"],
    long_description=read("README.md"),
)
