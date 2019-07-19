#import torchvision.models as models
import torch
from thop import profile

#print(torch.__version__)
#model = models.inception_v3()

from CenterNet52 import model as hourglass52
from CenterNet104 import model as hourglass104
from HRNet import model as hrnet 
from DLANet import model as dlanet
from HRSqueezeNet import model as hrsqueezenet

if __name__ == "__main__":
    #input = torch.randn(1, 3, 224, 224)
    input = torch.randn(1, 3, 511, 511)

    hourglass_52 = hourglass52()
    flops_hourglass52, params_hourglass52 = profile(hourglass_52, inputs = (input,))
    hourglass_104 = hourglass104()
    flops_hourglass104, params_hourglass104 = profile(hourglass_104, inputs = (input,))

    hr_net = hrnet()
    flops_hrnet, params_hrnet = profile(hr_net, inputs = (input,))
    dla_net = dlanet()
    flops_dlanet, params_dlanet = profile(dla_net, inputs = (input,))
    hr_squeeze_net = hrsqueezenet()
    flops_hrsqueezenet, params_hrsqueezenet = profile(hr_squeeze_net, inputs = (input,))

    print("hourglass52:")
    print("params:", params_hourglass52)
    print("flops:", flops_hourglass52)
    print("hourglass104:")
    print("params:", params_hourglass104)
    print("flops:", flops_hourglass104)
    print("hrnet:")
    print("params:", params_hrnet)
    print("flops:", flops_hrnet)
    print("dlanet34:")
    print("params:", params_dlanet)
    print("flops:", flops_dlanet)
    print("hr_squeeze_net:")
    print("params:", params_hrsqueezenet)
    print("flops:", flops_hrsqueezenet)


