import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from KernelWakeSleep import weights_init

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Identity(nn.Module):
    def __init__(self, *arg, **argv):
        super().__init__()
    def forward(self, x, *args, **argv):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.shape[0], *self.shape)

def large_dc_gen(nz=100, ngf=64, nc=1, bn=False, tanh=False, sigmoid=False, bias = False):
    norm_layer = nn.BatchNorm2d if bn else Identity
    layers = \
        [
        # input is Z, going into a convolution
        nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=bias),
        norm_layer(ngf * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        norm_layer(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 4 x 4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        norm_layer(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
        norm_layer(ngf * 1),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 1, nc     , 4, 2, 1, bias=False),
        
        # norm_layer(ngf),
        # nn.ReLU(True),
        # # state size. (ngf) x 32 x 32
        # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        # nn.Tanh()
        # # state size. (nc) x 64 x 64
        
        ]

    if tanh:
        layers += [nn.Tanh()]
    if sigmoid:
        layers += [nn.Sigmoid()]
    
    gen = nn.Sequential(*layers)
    gen.apply(weights_init)
    return gen

def dc_gen(nz=100, ngf=64, nc=1, bn=False, tanh=False, sigmoid=False):
    norm_layer = nn.BatchNorm2d if bn else Identity
    layers = \
        [
        # input is Z, going into a convolution
        nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
        norm_layer(ngf * 4),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        norm_layer(ngf * 2),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
        norm_layer(ngf * 1),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(ngf * 1, nc     , 4, 2, 1, bias=False),
        
        # norm_layer(ngf),
        # nn.ReLU(True),
        # # state size. (ngf) x 32 x 32
        # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        # nn.Tanh()
        # # state size. (nc) x 64 x 64
        
        ]

    if tanh:
        layers += [nn.Tanh()]
    if sigmoid:
        layers += [nn.Sigmoid()]
    
    gen = nn.Sequential(*layers)
    gen.apply(weights_init)
    return gen

def gan_gen(nz=100, ngf=64, nc=1, bn=False, tanh=False, sigmoid=False):

    norm_layer = nn.BatchNorm2d if bn else Identity

    layers = [
        Reshape((nz,)),
        nn.Linear(nz, ngf*2*8*8),
        Reshape((ngf*2,8,8)),
        norm_layer(ngf*2),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
        norm_layer(ngf*2, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
        norm_layer(ngf, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
    ]

    if tanh:
        layers += [nn.Tanh()]
    if sigmoid:
        layers += [nn.Sigmoid()]
    
    gen = nn.Sequential(*layers)
    gen.apply(weights_init)
    return gen

def fc_gen(n0,n1,n2,n3=None, tanh=False, sigmoid=False, softplus=False, nl_type="relu"):
    
    if nl_type == "relu":
        nl_layer = nn.ReLU
    elif nl_type == "softplus":
        nl_layer = nn.Softplus
    elif nl_type == "tanh":
        nl_layer = nn.Tanh
    else:
        raise(NameError, "layer type not right")
    layers = [
        Flatten(),
        nn.Linear(n0, n1),
        nl_layer(),
        nn.Linear(n1, n2)]
    if n3 is not None:
        layers += [nl_layer(), nn.Linear(n2,n3)]

    if tanh:
        layers += [nn.Tanh()]
    if sigmoid:
        layers += [nn.Sigmoid()]
    if softplus:
        layers += [nn.Softplus()]
    

    gen = nn.Sequential(*layers)
    gen.apply(weights_init)
    return gen

def lin_gen(n0,n1):
    
    layers = [
        nn.Flatten(),
        nn.Linear(n0, n1, bias=False)]

    gen = nn.Sequential(*layers)
    return gen
