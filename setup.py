import os
from setuptools import setup

# borrowed from https://pythonhosted.org/an_example_pypi_project/setuptools.html


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except FileNotFoundError:
        return ""


setup(
    name="sos_jax",
    version="0.2.0",
    author="Robert Dyro",
    description=(
        "Second Order Sensitivity Analysis for Bilevel Programming for JAX"
    ),
    license="MIT",
    packages=["sos_jax"],
    long_description=read("README.md"),
)
