import torch.nn as nn
import torchvision.models as models
from .td_helper import init, make_standard_block
from .FC3D import FC3D

class FC(nn.Module):
    def __init__(self, use_bn=True):  # Original implementation doesn't use BN
        super(FC, self).__init__()
        if use_bn:
            #vgg = models.vgg19(pretrained=True)
            vgg = FC3D()
            layers_to_use = list(list(vgg.children())[0].children())[:23]
        else:
            #vgg = models.vgg19_bn(pretrained=True)
            vgg = FC3D()
            layers_to_use = list(list(vgg.children())[0].children())[:33]
        self.vgg = nn.Sequential(*layers_to_use)
        # self.feature_extractor = nn.Sequential(make_standard_block(512, 256, 3),
        #                                        make_standard_block(256, 128, 3))
        self.feature_extractor = nn.Sequential(make_standard_block(32, 256, 3),
                                               make_standard_block(256, 128, 3))
        init(self.feature_extractor)

    def forward(self, x):
        x = self.vgg(x)
        x = self.feature_extractor(x)
        return x
