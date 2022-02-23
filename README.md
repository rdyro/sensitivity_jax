# sensitivity_jax

``sensitivity_jax`` is a package designed to allow taking first- and
**second-order** derivatives through optimization or any other fixed-point
process.

This package builds on top of [JAX](https://github.com/google/jax). We also
maintain an implementation in [PyTorch](https://pytorch.org/)
[here](https://rdyro.github.io/sensitivity_torch/).

## Documentation

[Documentation can be found here.](https://rdyro.github.io/sensitivity_jax/)

## Installation

Install from source
```bash
$ pip install git+https://github.com/rdyro/sensitivity_jax.git
```
or
```bash
$ git clone git@github.com:rdyro/sensitivity_jax.git
$ cd sensitivity_jax
$ python3 setup.py install --user
```

### Testing

Run all unit tests using
```bash
$ python3 setup.py test
```
