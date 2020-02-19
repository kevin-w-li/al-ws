import numpy as np
import torch
import torch.nn as nn
from math import pi
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import Likelihoods

def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.
    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.
    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.
    num_samples : int
        Number of samples of the function contained in dataset.
    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100, seed = None, noise_std = 0.1):

        np.random.seed(seed)
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        self.clean_data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.clean_data.append((x,y))
            y = y + torch.randn_like(y) * noise_std
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class NeuralProcess(nn.Module):
    
    min_std = 0.1
    
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        
        
        xy_to_r_layers = [nn.Linear(x_dim + y_dim, h_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(h_dim, h_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(h_dim, h_dim),]
        self.xy_to_r = nn.Sequential(*xy_to_r_layers)

        # r to q, this r is the aggregated version of the individual r's for each pair in context
        # from MuSigmaEncoder
        self.r_to_rhidden = nn.Sequential(
            nn.Linear(r_dim, r_dim), 
            nn.ReLU(True),
            nn.Linear(r_dim, r_dim), 
            nn.ReLU(True)
        )
        
        self.rhidden_to_mu     = nn.Linear(r_dim, z_dim)
        self.rhidden_to_sigma  = nn.Linear(r_dim, z_dim)

        # xz_to_pred
        # from Decoder
        xz_to_hidden_layers = [nn.Linear((x_dim + z_dim), h_dim), 
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),]
        self.xz_to_yhidden = nn.Sequential(*xz_to_hidden_layers)
        self.yhidden_to_mu = nn.Linear(h_dim, y_dim)
        self.yhidden_to_sigma = nn.Linear(h_dim, y_dim)


    @property
    def model_parameters(self):
        return self.parameters()
        
    def sample_logp(self, x_context, y_context, x_target, nsample):
        
        q_context = self.context_to_q(x_context, y_context)
        return self.q_to_samples(q_context, x_target, nsample) 
    
    def sample(self, x_context, y_context):
        
        q_context = self.context_to_q(x_context, y_context)
        return self.q_to_samples(q_context)[0]

    
    def context_to_q(self, x, y):
        r = self.xy_to_r(torch.cat([x, y], -1) )
        r = r.mean(0)
        h = self.r_to_rhidden(r)
        mu = self.rhidden_to_mu(h)
        pre_sigma = self.rhidden_to_sigma(r)
        sigma = self.min_std + (1-self.min_std) * torch.sigmoid((pre_sigma))
        q_context = td.Normal(mu, sigma)

        return q_context

    def q_to_samples(self, q_context, x_target, n_samples):

        '''
        This is for amortised learning
        draw n_samples samples of the function value at x_target given a q_context
        '''
        z_sample = q_context.sample([n_samples])
        logp0 = q_context.log_prob(z_sample).sum(-1)

        x_target = x_target.unsqueeze(0).repeat(z_sample.shape[0], 1,1)
        z_sample= z_sample.unsqueeze(1).repeat(1, x_target.shape[1], 1)
        h = self.xz_to_yhidden(torch.cat([x_target, z_sample], -1))
        y_mu = self.yhidden_to_mu(h).view(-1, self.y_dim)
        pre_y_sigma = self.yhidden_to_sigma(h).view(-1, self.y_dim)
        y_sigma = self.min_std + (1-self.min_std) * F.softplus(pre_y_sigma)
        p_y_pred = td.Normal(y_mu, y_sigma)
        y_sample = p_y_pred.sample().reshape(n_samples, -1)

        nat = Likelihoods.gaussian_nat(p_y_pred).view(n_samples, -1, 2)
        nat = torch.cat([nat[...,0], nat[...,1]], -1)
        norm = Likelihoods.gaussian_norm(p_y_pred).view(n_samples,-1).sum(-1)
        norm = norm - logp0

        return y_sample, nat, norm


    def q_to_y_dist(self, q, x_target):

        '''
        this is used in NP's original learning method
        '''
        z_sample = q.rsample([1])
        z_sample = z_sample.repeat(x_target.shape[0],1)

        h = self.xz_to_yhidden(torch.cat([x_target, z_sample], -1))
        y_mu = self.yhidden_to_mu(h)
        pre_y_sigma = self.yhidden_to_sigma(h)
        y_sigma = self.min_std + (1-self.min_std) * F.softplus(pre_y_sigma)
        p_y_pred = td.Normal(y_mu.view(-1, self.y_dim), y_sigma)

        return p_y_pred


    def posterior(self, x_context, y_context, x_target, n_samples=100, ax=None):
        '''
        return posterior given context pair and x_target
        '''
        q_context = self.context_to_q(x_context, y_context)

        z_sample = q_context.sample([n_samples])

        x_target = x_target.unsqueeze(0).repeat(z_sample.shape[0], 1,1)
        z_sample= z_sample.unsqueeze(1).repeat(1, x_target.shape[1], 1)
        h = self.xz_to_yhidden(torch.cat([x_target, z_sample], -1))
        mu = self.yhidden_to_mu(h).detach()

        return mu
