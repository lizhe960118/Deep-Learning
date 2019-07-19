import torch
import torch.nn as nn

from py_utils import DLASeg, convolution, residual
 
class model(DLASeg):

    def __init__(self):
        down_ratio = 4
        out_dim = 80

        super(model, self).__init__(
            "dla34",
            pretrained = None, 
            down_ratio=down_ratio,
            final_kernel=1,
            last_level=5,
            out_dim = 80,
            cnv_dim=256
        )
