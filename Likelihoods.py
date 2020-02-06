import torch
import numpy as np

def gaussian_nat(dist):
    # takes a gaussian distriubiton and observation to compute a list of natural parameters
    # first element corresponds to x, second to x**2
    return torch.cat([p.reshape(dist.mean.shape[0], -1) for p in dist._natural_params], -1)

def gaussian_suff(x):
    return torch.cat([x, x**2], -1)
   
def gaussian_norm(dist):
    mu = dist.mean.reshape(dist.mean.shape[0], -1)
    std  = dist.stddev.reshape(dist.mean.shape[0], -1)
    ns  = 0.5 * torch.sum((mu**2) / (std**2) + torch.log(2*np.pi*std**2), -1)
    return ns

def loggaussian_nat(dist):
    nat = dist.base_dist._natural_params
    nat = (nat[0] - 1.0, nat[1])
    return torch.cat([p.reshape(dist.mean.shape[0], -1) for p in nat], -1)

def loggaussian_suff(x):
    return torch.cat([x.log(), x.log()**2], -1)
   
def loggaussian_norm(dist):
    mu  = dist.base_dist.mean.reshape(dist.mean.shape[0], -1)
    std = dist.base_dist.stddev.reshape(dist.mean.shape[0], -1)
    ns  = 0.5 * torch.sum((mu**2) / (std**2) + torch.log(2*np.pi*std**2), -1)
    return ns

def poisson_nat(dist):
    return dist._natural_params()

def poisson_norm(dist):
    return dist._natural_params(*dist._natural_params)

def poisson_suff(x):
    return x


class FullGaussianObservation(object):

    def __init__(self, mean_network, std_network):

        super(FullGaussianObservation, self).__init__()

        self.mean_network = mean_network
        self.std_network  = std_network
        
    def conditional_param(self, z):
        
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        mu = self.mean_network(z)
        std = self.std_network(z)
        dist = torch.distributions.normal.Normal(mu, std)
        return dist

    def sample(self, z, detach=True):
        
        if detach:
            with torch.no_grad():
                dist = self.conditional_param(z)
            return dist.sample([]).reshape(z.shape[0], -1)
        else:
            dist = self.conditional_param(z)
            return dist.rsample([]).reshape(z.shape[0], -1)

    def logp(self, z, x):

        dist = self.conditional_param(z)
        shape = dist.mean.shape
        l = dist.log_prob(x.reshape(*shape)).sum(-1)
        if len(l.shape)==3:
            return l.sum(-1).sum(-1)
        else:
            return l
        
        
    def norm(self, z):
        dist = self.conditional_param(z)
        return gaussian_norm(dist)

    def nat(self, z):
        dist  = self.conditional_param(z)
        return gaussian_nat(dist)
    
    def suff(self, x):
        return gaussian_suff(x)
    

class GaussianObservation(object):

    def __init__(self, network, log_std):

        super(GaussianObservation, self).__init__()

        self.log_std = log_std
        self.network = network
        
    def conditional_param(self, z):
        
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        mu = self.network(z)
        if self.log_std.ndim != 0:
            std = self.log_std.exp().reshape(*mu.shape[1:])
        else:
            std = self.log_std.exp()
        dist = torch.distributions.normal.Normal(mu, std)
        return dist

    def sample(self, z, detach=True):
        
        if detach:
            with torch.no_grad():
                dist = self.conditional_param(z)
            return dist.sample([]).reshape(z.shape[0], -1)
        else:
            dist = self.conditional_param(z)
            return dist.rsample([]).reshape(z.shape[0], -1)

    def logp(self, z, x):

        dist = self.conditional_param(z)
        shape = dist.mean.shape
        l = dist.log_prob(x.reshape(*shape)).sum(-1)
        if len(l.shape)==3:
            return l.sum(-1).sum(-1)
        else:
            return l
        
        
    def norm(self, z):
        std = self.log_std
        mu  = self.conditional_param(z).mean.reshape(z.shape[0], -1)
        ns  = 0.5 * torch.sum((mu**2) / torch.exp(2*std) + torch.log(2*np.pi*torch.exp(2*std)), -1)
        return ns

    def nat(self, z):
        dist  = self.conditional_param(z)
        return torch.cat([p.reshape(z.shape[0], -1) for p in dist._natural_params], -1)
    
    def suff(self, x):
        return torch.cat([x, x**2], -1)
    
class BernoulliObservation(object):
    
    def __init__(self, network, eps = 1e-4):
        
        self.network = network
        self.eps = eps
        
    def conditional_param(self, z):
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        z = self.network(z) * (1-2*self.eps) + self.eps
        #z = self.network(z).clamp(self.eps, 1-self.eps)
        return z.reshape(z.shape[0], -1)
        
    def sample(self, z, detach=True):
        
        with torch.no_grad():
            mu = self.conditional_param(z)
            #mu.sigmoid_()
            s = torch.floor(torch.rand_like(mu)+mu)
        return s.reshape(z.shape[0], -1)
    
    def norm(self, z):
        mu = self.conditional_param(z)
        #mu = torch.where(mu > 15, mu, torch.log(1+mu.exp()))
        #return mu.sum(-1)
        return -torch.log(1-mu).sum(-1)
    
    def nat(self, z):
        mu = self.conditional_param(z)
        mu = torch.log(mu/(1-mu))
        return mu
    
    def suff(self, x):
        return x

    def logp(self, z, x, detach=False):
        if detach:
            with torch.no_grad:
                mu = self.conditional_param(z)
        else: 
            mu = self.conditional_param(z)
        return (x * mu.log() + (1-x)*(1-mu).log() ).sum(-1)

    def sample_nat_norm(self, z):

        with torch.no_grad():
            mu = self.conditional_param(z)
            #mu.sigmoid_()
            s = torch.floor(torch.rand_like(mu)+mu)
            nat = torch.log(mu/(1-mu))
            norm = -torch.log(1-mu).sum(-1)
        return s.reshape(z.shape[0], -1), nat, norm


class PoissonObservations(object):
    
    def __init__(self, rate_fun):
        
        self.rate_fun = rate_fun

    def conditional_param(self, z):
        mu = self.rate_fun(z)
        return mu

    def sample(self, z):
        mu = self.conditional_param(z)
        return torch.poisson(mu)

    def sample_nat_norm(self, z):
        mu = self.conditional_param(z)
        x = torch.poisson(mu)
        nat = mu.log()
        norm = -mu.sum(-1)
        return x, nat, norm

    def nat(self, z):
        mu = self.conditional_param(z)
        return mu.log()

    def norm(self, z):
        mu = self.conditional_param(z)
        return -mu.sum(-1)
        
    def suff(self, x):
        return x
