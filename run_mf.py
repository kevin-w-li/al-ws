import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from math import pi
from torch.utils.data import DataLoader
from importlib import reload
import torch.distributions as td
import torch.nn.functional as F
import torch.optim as opt
from KernelWakeSleep import KRR_2, Kernel, estimate_median_distance, weights_init
import Likelihoods,tqdm.notebook
from Likelihoods import BernoulliObservation
import pdb
from Generators import Flatten
from itertools import chain
from torchvision import datasets, transforms
from torch.utils.data import Subset
import Features
from tqdm import tqdm
import Data
from utils import logit, Binarize
import argparse

        
class MF_Gen(nn.Module):
    
    def __init__(self, k, d, device, wmin=0.01, wmax=0.03):
        """
        beta-gamma matrix factorization model
        args:
         - k = latent dimension
         - d = data dimension
         - device = cpu/gpu unit for params
        """
        super().__init__()
        self.k = k
        prior = td.Uniform(wmax, wmax)
        W_init = prior.sample(sample_shape=[k, d]).to(device)
        self.log_W = nn.Parameter(W_init.log(),
                                  requires_grad=True)
        self.b = nn.Parameter(0.1 * torch.ones(d), requires_grad=True)
        self.device=device
        
    def forward(self, z):
        """
        generate the mean parameter for the Bernoulli distribution
        """
        z = z.reshape(-1, self.k)
        z = z.to(self.device)
        p = (torch.mm(logit(z), self.log_W.exp()) + self.b).sigmoid()
        return p

    def log_prob(self, x, z):
        """
        return the log-joint probability
        """
        p = self.forward(z)
        px = td.Bernoulli(probs=p)
        return px.log_prob(x).sum(-1)

class DEF_Gen(nn.Module):
    #TODO 
    
    def __init__(self, k1, k2, k3, d, device, wmin=0.01, wmax=0.03):
        """
        beta-gamma matrix factorization model
        args:
         - k = latent dimension
         - d = data dimension
         - device = cpu/gpu unit for params
        """
        super().__init__()
        prior = td.Uniform(wmax, wmax)
        W_init = prior.sample(sample_shape=[k, d]).to(device)
        self.log_W = nn.Parameter(W_init.log(),
                                  requires_grad=True)
        self.b = nn.Parameter(0.1 * torch.ones(d), requires_grad=True)
        self.device=device
        
    def forward(self, z):
        """
        generate the mean parameter for the Bernoulli distribution
        """
        z = z.view(-1, self.k)
        z = z.to(self.device)
        p = (torch.mm(logit(z), self.log_W.exp()) + self.b).sigmoid()
        return p

def train(device, batch_size, train_loader,\
        n_sleep=1500, n_epochs=30, save=True, n_kernel_nets=1):
    # prior on z
    p_z = td.Uniform(0.0, 1.0)
    # latent dimension
    k = 100
    # data dimension (MNIST)
    d = 784
    # num train
    n_train =  len(train_loader) * batch_size

    # define generative model
    mf_network = MF_Gen(k, d, device).to(device)
    bern_obs = BernoulliObservation(mf_network)

    z_s = p_z.sample(sample_shape=[batch_size, k]).to(device)
    # optionally train kernel with cross validation
    x_s, nat, norm = bern_obs.sample_nat_norm(z_s.to(device))


    # could try a linear network here
    kernel_network = [torch.nn.Identity()]
    block_size = 100
    # kernel_network = [Features.lin_feat(d, k) for i in range(n_kernel_nets)]
    kernel_networks = []
    Fdata = []
    for i in range(n_kernel_nets):
        
        # no network
        # feat = torch.nn.Identity().double()
        
        # linear network followed by batch norm (in the papaer)
        feat = Features.lin_feat(d, block_size, bn=True).to(device)

        # convolutional network followed by batch norm
        # k = Features.dc_feat(image_size,ngf, nfinal=100, bn=False, nl=True, nc=nc).to(device)
        feat.apply(weights_init)
        
        Fdata += feat(x_s),
        kernel_networks += feat,
        
    Fdata   = torch.cat(Fdata, -1) 
    Fmedian =  estimate_median_distance(Fdata)

    kernel = Kernel(kernel_networks, Fmedian, train_sigma=True, train_lam=True).to(device)


    # define optimizer
    optimizer = opt.Adam(mf_network.parameters(), lr=1e-3) # 1e-3 is unstable, 3e-4 also works
    kernel_optimizer = opt.Adam(kernel.parameters(), lr=1e-3)

    # prior on w
    alpha = 0.9; beta = 0.3; 
    W_prior = td.Gamma(alpha, beta * torch.ones((), device=device))

    # keep track the cross-validation losses
    losses = []
    # keep track the Elogp objectives
    Elogps = []

    for ei in range(1, n_epochs+1):
        for di, (x,) in enumerate(tqdm((train_loader))):

            # generate data
            #x, _ = data
            x = x.float().to(device)
            # binarize
           #x = x / x.max() 
            #x = x.round()
            x = x.view(batch_size, -1)

            #if di % 1 == 0:
            # optionally train kernel with cross validation
            z_s = p_z.sample(sample_shape=[n_sleep, k])
            x_s, nat, norm = bern_obs.sample_nat_norm(z_s)
            
            z_s = p_z.sample(sample_shape=[150, k])
            x_s_val, nat_val, norm_val = bern_obs.sample_nat_norm(z_s)

            Elogp_val = KRR_2(bern_obs.suff, kernel, x_s_val, x_s,
                            norm, nat)  
            Elogp_true = (bern_obs.suff(x_s_val) * nat_val).sum(-1) - norm_val
            loss = torch.mean((Elogp_val - Elogp_true)**2)
            losses += loss.detach().cpu().numpy(),
            kernel_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            kernel_optimizer.step()

            # draw a z from the prior 
            z_s = p_z.sample(sample_shape=[n_sleep, k])
            
            # generate x, get natural param and log normalizer
            x_s, nat, norm = bern_obs.sample_nat_norm(z_s)
            x_s = x_s.detach()
            
            Elogp = KRR_2(bern_obs.suff, kernel, x, x_s,
                    norm, nat).mean()
        
            # incorporate prior into loss 
            W = mf_network.log_W.exp()
            loss = -Elogp - W_prior.log_prob(W + 1e-15).sum().to(device) / n_train
            if di % 49 == 0: print ("epoch {}/{}, iter {}: loss = {}".format(ei, n_epochs, di, loss)); 

            Elogps += Elogp.detach().cpu().numpy(),
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if save and ei % 30 == 0:
            print ('saving checkpoint...')
            sig = "_nnets{}_nsleep{}".format(n_kernel_nets, n_sleep)
            np.savetxt('../ALWS_saved/mf/samples/raw_samples{}{}.csv'.format(ei, sig), x_s.detach().cpu().numpy()[:25,:], delimiter=',')
            np.savetxt('../ALWS_saved/mf/ws/w{}{}.csv'.format(ei, sig), W.detach().cpu().numpy(), delimiter=',')
            np.savetxt('../ALWS_saved/mf/bs/b{}{}.csv'.format(ei, sig), mf_network.b.detach().cpu().numpy(), delimiter=',')
            np.savetxt('../ALWS_saved/mf/kernel_loss{}.csv'.format(sig), np.array(losses), delimiter=',')
            np.savetxt('../ALWS_saved/mf/elogp{}.csv'.format(sig), np.array(Elogps), delimiter=',')
            torch.save(mf_network.state_dict(), "../ALWS_saved/mf/models/mf{}{}.pt".format(ei, sig))
            print ('done.')

def main(n_train=5000, n_sleep=1500, n_epochs=30, save=True, n_kernel_nets=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using GPU: ', torch.cuda.is_available())
    print('starting training...')
    # set up data
    batch_size = 100
    # train data 
    """
    train_data = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            Binarize()
                        ]))
    train_data = Subset(train_data, list(range(n_train)))

    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size, shuffle=True)
    """

    dname = "mnist"
    binary= True 
    nc = 1

    train_loader, _ = Data.load_data(dname, binary=binary, ntrain=n_train, image_size=28, batch_size=batch_size)



    train(device, batch_size, train_loader,\
        n_sleep=n_sleep, n_epochs=n_epochs, save=save, n_kernel_nets=n_kernel_nets)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    def str2bool(s):
        if s == "False": return False;
        if s == "True": return True;
        else: raise ValueError("Improper boolean argument"); 

    parser.add_argument('--save', nargs='?', const=1, type=bool, default=True)
    parser.add_argument('--n_sleep', nargs='?', const=1, type=int, default=1500)
    parser.add_argument('--n_epochs', nargs='?', const=1, type=int, default=20)
    parser.add_argument('--n_kernel_nets', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--n_train', nargs='?', const=1, type=int, default=5000)

    args = parser.parse_args()
    main(n_train=args.n_train, n_sleep=args.n_sleep, n_epochs=args.n_epochs,\
        save=args.save, n_kernel_nets=args.n_kernel_nets)