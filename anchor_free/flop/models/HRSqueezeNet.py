import torch
import torch.nn as nn

from py_utils import hr_squeeze, convolution, residual

class model(hr_squeeze):
    def __init__(self):
        out_dim = 80
        super(model, self).__init__(
            out_dim,cnv_dim=256
        )