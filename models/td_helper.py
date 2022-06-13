import torch.nn as nn
import math


def init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True):
    layers = []
    layers += [nn.Conv3d(feat_in, feat_out, kernel, stride, padding)]
    if use_bn:
        layers += [nn.BatchNorm3d(feat_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def make_2019_block(feat_in, feat_out, kernel):
    block1 = make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True)
    block2 = make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True)
    block3 = make_standard_block(feat_in, feat_out, kernel, stride=1, padding=1, use_bn=True)
    return block1, block2, block3
