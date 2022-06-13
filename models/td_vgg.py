import torch.nn as nn
import torchvision.models as models
from models.td_helper import init, make_standard_block
from models.VGG3D import VGG3D


class VGG(nn.Module):
    def __init__(self, use_bn=True):  # Original implementation doesn't use BN
        super(VGG, self).__init__()
        ch_pick_layer = 2
        if use_bn:
            #vgg = models.vgg19(pretrained=True)
            vgg = VGG3D()
            #layers_to_use = list(list(vgg.children())[0].children())[:23]
            layers_to_use = []
            for i in range(ch_pick_layer):
                tmp = list(vgg.children())[i]
                for j in range(len(tmp)):
                    layers_to_use.append(tmp[j])
                del tmp
        else:
            #vgg = models.vgg19_bn(pretrained=True)
            vgg = VGG3D()
            #layers_to_use = list(list(vgg.children())[0].children())[:33]
            layers_to_use = []
            for i in range(ch_pick_layer): # I pick to use 10 layers from the in house vgg3D
                #print('i=', i)
                tmp = list(vgg.children())[i]
                for j in range(len(tmp)):
                    layers_to_use.append(tmp[j])
                del tmp

        self.vgg = nn.Sequential(*layers_to_use)
        # self.feature_extractor = nn.Sequential(make_standard_block(512, 256, 3),
        #                                        make_standard_block(256, 128, 3))

        #vggPT = models.vgg19_bn(pretrained=True) # chin just test something
        #layers_to_usePT = list(list(vggPT.children())[0].children())[:33]

        # self.feature_extractor = nn.Sequential(make_standard_block(32, 256, 3),
        #                                        make_standard_block(256, 128, 3))

        # self.feature_extractor = nn.Sequential(make_standard_block(32, 128, 3),
        #                                        make_standard_block(128, 128, 3))

        self.feature_extractor = nn.Sequential(make_standard_block(32, 64, 3),
                                               make_standard_block(64, 128, 3))

        init(self.feature_extractor)

    def forward(self, x):
        x = self.vgg(x)
        print(x.shape)
        x = self.feature_extractor(x)
        print(x.shape)
        return x
