import numpy as np, torch

from jfi import jaxm

from ..utils import j2n

t2n = lambda x: x.detach().cpu().numpy()
n2t = lambda x, **kw: torch.as_tensor(np.array(x), **kw)
j2t = lambda x, **kw: torch.as_tensor(np.array(x), **kw)
x2t = lambda x, **kw: torch.as_tensor(
    x if not isinstance(x, jaxm.jax.Array) else j2n(x), **kw
)
x2j = lambda x: jaxm.array(x if not isinstance(x, torch.Tensor) else t2n(x))
x2n = lambda x: np.array(x) if not isinstance(x, torch.Tensor) else t2n(x)
