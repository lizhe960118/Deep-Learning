import torch.nn as nn
from py_utils import kp, convolution, residual


def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(kp):
    def __init__(self):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 80

        super(model, self).__init__(
             n, 1, dims, modules, out_dim,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )
