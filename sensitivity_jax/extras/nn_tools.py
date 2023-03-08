import pdb, time
from collections import OrderedDict as odict
from functools import partial

import torch, numpy as np

from .. import jax_friendly_interface as jfi

jaxm = jfi.init()

from ..utils import t2j


# utilities ####################################################################
def nn_all_params(nn):
    return t2j(
        torch.cat([param.detach().reshape(-1) for param in nn.parameters()])
    ).astype(jaxm.ones(()).dtype)


# main forward network via functional interface ################################
def linear(input, *args, **kwargs):
    W, b = args
    return input @ W.T + b


def conv(input, *args, **kwargs):
    C, C0 = args
    return (
        jaxm.lax.conv(input, C, kwargs["stride"], "VALID") + C0[..., None, None]
    )


def tanh(input, *args, **kwargs):
    return jaxm.nn.tanh(input)


def softmax(input, *args, **kwargs):
    return jaxm.nn.softmax(input)


def relu(input, *args, **kwargs):
    return jaxm.nn.relu(input)


def flatten(input, *args, **kwargs):
    bshape = input.shape[0]
    return input.reshape((bshape, -1))


NN2NAME_MAP = {
    torch.nn.Tanh: "tanh",
    torch.nn.Softmax: "softmax",
    torch.nn.ReLU: "relu",
    torch.nn.Flatten: "flatten",
    torch.nn.Linear: "linear",
    torch.nn.Conv1d: "conv",
    torch.nn.Conv2d: "conv",
    torch.nn.Conv3d: "conv",
}

NAME2FN_MAP = {
    "tanh": tanh,
    "softmax": softmax,
    "relu": relu,
    "flatten": flatten,
    "linear": linear,
    "conv": conv,
}

NN2KWARGS_MAP = {
    torch.nn.Tanh: lambda x: dict(),
    torch.nn.ReLU: lambda x: dict(),
    torch.nn.Flatten: lambda x: dict(),
    torch.nn.Linear: lambda x: dict(),
    torch.nn.Softmax: lambda x: dict(dim=x.dim),
    torch.nn.Conv1d: lambda x: dict(
        stride=x.stride,
        padding=x.padding,
        dilation=x.dilation,
        groups=x.groups,
    ),
    torch.nn.Conv2d: lambda x: dict(
        stride=x.stride,
        padding=x.padding,
        dilation=x.dilation,
        groups=x.groups,
    ),
    torch.nn.Conv3d: lambda x: dict(
        stride=x.stride,
        padding=x.padding,
        dilation=x.dilation,
        groups=x.groups,
    ),
}
################################################################################


################################################################################
def nn_forward_gen(nn, debug=False):
    assert isinstance(nn, torch.nn.Sequential)
    secs = jaxm.cumsum(
        jaxm.array(
            [param.detach().numel() for param in nn.state_dict().values()]
        )
    )
    secs = np.array([float(sec) for sec in secs])
    shapes = [tuple(param.shape) for param in nn.state_dict().values()]
    counts = [len(list(mod.parameters())) for mod in list(nn.modules())[1:]]
    mod_names = [NN2NAME_MAP[type(mod)] for mod in list(nn.modules())[1:]]
    layer_kw = [NN2KWARGS_MAP[type(mod)](mod) for mod in list(nn.modules())[1:]]

    def fn(X, params):
        layers = jaxm.split(params, secs)
        Z, k = X, 0
        for (c, name, kw) in zip(counts, mod_names, layer_kw):
            layer = [
                param.reshape(shape)
                for (param, shape) in zip(layers[k : k + c], shapes[k : k + c])
            ]
            Z = NAME2FN_MAP[name](Z, *layer, **kw)
            k = k + c
        return Z

    return fn


################################################################################


################################################################################
def flax_forward_gen(model, x):
    state = model.init(jfi.key, x)
    all_params, param_struct = jaxm.jax.tree_flatten(state)
    shapes = [z.shape for z in all_params]
    sizes = [z.size for z in all_params]
    secs = np.array(jaxm.cumsum(jaxm.array(sizes)))
    all_params = jaxm.cat([param.reshape(-1) for param in all_params], -1)

    def fwd_fn(all_params, x):
        all_params = jaxm.split(all_params, secs)
        all_params = [
            param.reshape(shape) for (param, shape) in zip(all_params, shapes)
        ]
        all_params = jaxm.jax.tree_unflatten(param_struct, all_params)

        yp = model.apply(all_params, x)
        return yp

    return fwd_fn, all_params


################################################################################
