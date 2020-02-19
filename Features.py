import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.shape[0], *self.shape)

class Identity(nn.Module):
    def __init__(self, *arg, **argv):
        super().__init__()
    def forward(self, x):
        return x

def large_dc_feat(image_size=64, ndf = 32, nc=1, nfinal = None, nl=True, bn=False):
    norm_layer = nn.BatchNorm2d if bn else nn.Identity
    nl_layer   = nn.ReLU if nl else nn.Identity
    layers = \
        [   
            Reshape((nc, image_size, image_size)),
            # state size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 1),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 2),

            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 4),

            # state size. (ndf*1) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 8),

            Flatten()
        ]

    if nfinal is not None:
        Df = 128 * ndf
        layers += [nn.Linear(Df, nfinal), nl_layer(inplace=True), nn.BatchNorm1d(nfinal)]
        # layers += [nn.Linear(Df, nfinal), nl_layer(inplace=True)]

    return nn.Sequential(*layers)

def dc_feat(image_size=32, ndf = 32, nc=1, nfinal = None, nl=True, bn=False, final_bn=True, final_nl=False):
    norm_layer = nn.BatchNorm2d if bn else nn.Identity
    nl_layer   = nn.ReLU if nl else nn.Identity
    layers = \
        [   
            Reshape((nc, image_size, image_size)),
            # state size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 1),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 2),

            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nl_layer(inplace=True),
            norm_layer(ndf * 4),

            Flatten()
        ]

    if nfinal is not None:
        Df = 64 * ndf
        layers += [nn.Linear(Df, nfinal)]
    if final_bn:
        layers += [nn.BatchNorm1d(nfinal)]

    if final_nl:
        layers += [final_nl()]

    return nn.Sequential(*layers)

def fc_feat(n0,n1,n2,n3=0,bn=True, final_bn=True, nl=True):
    
    norm_layer = nn.BatchNorm1d if bn else Identity
    nl_layer   = nn.Softplus if nl else Identity

    layers = [
        Flatten(),
        nn.Linear(n0, n1),
        norm_layer(n1),
        nl_layer(True),
        nn.Linear(n1, n2),
        norm_layer(n2),
        nl_layer(True),
    ]

    if n3 > 0:
        layers += [
            nn.Linear(n2, n3),
        ]
    if final_bn:
        layers += norm_layer(n3),

    return nn.Sequential(*layers)

def lin_feat(n0,n1,bn=True):
    
    norm_layer = nn.BatchNorm1d if bn else Identity

    layers = [
        Flatten(),
        nn.Linear(n0, n1),
        norm_layer(n1),
    ]

    return nn.Sequential(*layers)

def lin_fc_feat(n0,n1,bn=True):
    
    norm_layer = nn.BatchNorm1d if bn else Identity

    layers = [
        Flatten(),
        nn.Linear(n0, n1),
        norm_layer(n1),
    ]

    return nn.Sequential(*layers)

def lin_dc_feat(*args, **kwargs):
    
    class LinFcFeature(nn.Module):
        def __init__(self, *args, **kwargs):
            self.__init__()

            self.network = dc_feat(*args, **kwargs)

        def forward(self, x):
            return self.network(torch.cat([x, self.network(x)], -1))

    feat = LinFcFeature(*args, **kwargs)
    return feat
