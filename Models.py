import torch, math
import torch.nn as nn
import torch.distributions as td
import Generators, Likelihoods
from KernelWakeSleep import KRR_2, Kernel, SetKernel, estimate_median_distance, RFKernel
from itertools import chain
import torch.nn.functional as F

def weights_init(m, std=0.3):     
    
    classname = m.__class__.__name__   
    if classname.find('Conv') != -1: 
        m.weight.data.normal_(0.0, std) 
    elif classname.find('Linear') != -1:    
        m.weight.data.normal_(0.0, std)  
    elif classname.find('BatchNorm') != -1:  
        m.weight.data.normal_(1.0, std)   
        m.bias.data.fill_(0) 

class Models(nn.Module):
    def __init__(self):
        super().__init__()

class NeuralProcess(nn.Module):

    def __init__(self, device, batch_size=1, num_context=4, num_target=4, x_dim=1, y_dim=1, r_dim=50, z_dim=50, h_dim=50, kernel="set"):

        super().__init__()

        # define kernels
        """
        kernel_network = [torch.nn.Sequential(
                        torch.nn.Linear(num_context+num_target, h_dim),
                        torch.nn.ReLU(True),
                        torch.nn.Linear(h_dim,h_dim))]
        """

        assert kernel in ["set", "rf", "normal", None]
        kernel_network = [torch.nn.Identity()]
        
        if kernel == "set":
            # use a kernel over sets
            self.kernel = SetKernel(kernel_network, 1.0, train_sigma=True, train_lam=True).to(device)
        elif kernel == "rf":
            self.kernel = RFKernel(num_target, kernel_network, 1.0, nfeat = 100, train_sigma=True, train_lam=True).to(device)
        elif kernel == "normal":
            self.kernel = Kernel(kernel_network, 1.0, train_sigma=True, train_lam=True).to(device)
        elif kernel is None:
            pass

        else:
            raise(NameError, "kernele is invalid")
            

        ## define NP
        # xy_to_r
        xy_to_r_layers = [nn.Linear(x_dim + y_dim, h_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(h_dim, h_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(h_dim, h_dim),]
        self.xy_to_r = nn.Sequential(*xy_to_r_layers).to(device)

        # r to q, this r is the aggregated version of the individual r's for each pair in context
        self.r_to_rhidden = nn.Sequential(
            nn.Linear(r_dim, r_dim), 
            nn.ReLU(True),
            nn.Linear(r_dim, r_dim), 
            nn.ReLU(True)
        ).to(device)
        self.rhidden_to_mu     = nn.Linear(r_dim, z_dim).to(device)
        self.rhidden_to_sigma  = nn.Linear(r_dim, z_dim).to(device)

        # xz_to_y
        xz_to_hidden_layers = [nn.Linear((x_dim + z_dim), h_dim), 
                nn.ReLU(inplace=True),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(inplace=True),]
        self.xz_to_yhidden = nn.Sequential(*xz_to_hidden_layers).to(device)
        self.yhidden_to_mu = nn.Linear(h_dim, y_dim).to(device)
        self.yhidden_to_sigma = nn.Linear(h_dim, y_dim).to(device)

        # list of networks
        self.networks = [self.xy_to_r, self.r_to_rhidden, self.rhidden_to_mu, self.rhidden_to_sigma, 
                    self.xz_to_yhidden, self.yhidden_to_mu, self.yhidden_to_sigma]
        self.y_dim = y_dim

    @property
    def model_parameters(self):
        return chain(*[n.parameters() for n in self.networks])
    

    def context_to_q(self, x_context, y_context):
        r = self.xy_to_r(torch.cat([x_context, y_context], -1) )
        r = r.mean(0)
        h = self.r_to_rhidden(r)
        mu = self.rhidden_to_mu(h)
        pre_sigma = self.rhidden_to_sigma(r)
        sigma = 0.1 + 0.9 * torch.sigmoid((pre_sigma))
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
        y_sigma = 0.1 + 0.9 * F.softplus(pre_y_sigma)
        p_y_pred = td.Normal(y_mu, y_sigma)
        y_sample = p_y_pred.sample().reshape(n_samples, -1)

        nat = Likelihoods.gaussian_nat(p_y_pred).view(n_samples, -1, 2)
        nat = torch.cat([nat[...,0], nat[...,1]], -1)
        norm = Likelihoods.gaussian_norm(p_y_pred).view(n_samples,-1).sum(-1)
        norm = norm - logp0
        
        return y_sample, nat, norm, 

    def q_to_y_pred(self, q, x_target):
        '''
        this is used in NP's original learning method
        '''
        z_sample = q.rsample([1])
        z_sample = z_sample.repeat(x_target.shape[0],1)

        h = self.xz_to_yhidden(torch.cat([x_target, z_sample], -1))
        y_mu = self.yhidden_to_mu(h)
        pre_y_sigma = self.yhidden_to_sigma(h)
        y_sigma = 0.1 + 0.9 * F.softplus(pre_y_sigma)
        p_y_pred = td.Normal(y_mu.view(-1, y_dim), y_sigma)
        
        return p_y_pred

    def posterior(self, x_context, y_context, x_target, n_samples=100, ax=None):
        '''
        return posterior given context pair and x_target
        '''
        q_context  = self.context_to_q(x_context, y_context)

        z_sample = q_context.sample([n_samples])

        x_target = x_target.unsqueeze(0).repeat(z_sample.shape[0], 1,1)
        z_sample= z_sample.unsqueeze(1).repeat(1, x_target.shape[1], 1)
        h = self.xz_to_yhidden(torch.cat([x_target, z_sample], -1))
        mu = self.yhidden_to_mu(h).detach()
        
        return mu


class GMM(Models):
    
    def __init__(self, K, D):
        
        super().__init__()
        
        self.alpha = nn.Parameter(torch.ones(K), requires_grad=True)
        self.mu    = nn.Parameter(torch.randn(K, D)*0.2, requires_grad=True)
        self.chol  = nn.Parameter(torch.stack([torch.eye(D,D)*0.5]*K), requires_grad=True)
        self.dir   = td.Dirichlet(self.alpha)

        
    def sample_logp0(self, n):
        
        logp0 = 0
    
        pz_pi= td.Categorical(self.alpha.softmax(0))
        z    = pz_pi.sample([n])
        logp0 = logp0 + pz_pi.log_prob(z)
        
        mean = self.mu[z,:]
        cov  = torch.einsum("ijk,ilk->ijl", self.chol, self.chol)
        cov = cov + torch.eye(cov.shape[-1], device=cov.device)  * 1e-6
        cov = cov[z,...]
        
        py_z = td.MultivariateNormal(mean, cov)

        y    = py_z.sample([])

        norm = -py_z.log_prob(y)
        nat  = torch.zeros(y.shape[0], 2*2, device=norm.device)
    
        return (None, z, None, None, y), y, norm - logp0, nat

    
    def sample(self, n):
        return self.sample_logp0(n)[1]
    
    def suff(self, x):
        return Likelihoods.gaussian_suff(x)

class PinWheelGMM(nn.Module):
    
    def __init__(self, K, D, nh=10):
        
        super().__init__()
        
        self.alpha = nn.Parameter(torch.ones(K), requires_grad=True)
        self.mu    = nn.Parameter(torch.randn(K, D)*0.5, requires_grad=True)
        self.chol  = nn.Parameter(torch.stack([torch.eye(D,D)*0.3]*K), requires_grad=True)
        
        self.mean_network = Generators.fc_gen(D,nh,D, nl_type="relu")    
        self.mean_network.apply(weights_init)
        self.log_std = nn.Parameter(torch.zeros(D)-1, requires_grad=True)

        self.obs   = Likelihoods.GaussianObservation(self.mean_network, self.log_std)

        
    def sample_logp0(self, n):
        
        logp0 = 0
        
        #pz_pi= td.Categorical(self.alpha.softmax(0).clamp(1e-5,1-1e-5))
        pz_pi= td.Categorical(logits=self.alpha)
        z    = pz_pi.sample([n])
        logp0 = logp0 + pz_pi.log_prob(z)
        
        mean = self.mu[z,:]
        cov  = torch.einsum("ijk,ilk->ijl", self.chol, self.chol)
        cov = cov + torch.eye(cov.shape[-1], device=cov.device)  * 1e-6
        cov = cov[z,...]
        
        py_z = td.MultivariateNormal(mean, cov)
        y    = py_z.sample([])
        logp0 += py_z.log_prob(y)
                
        x    = self.obs.sample(y)
        nat  = self.obs.nat(y)
        norm = self.obs.norm(y)       
        
    
        return (None, z, None, None, y), x, norm - logp0, nat

    
    def sample(self, n):
        return self.sample_logp0(n)[1]

    def suff(self, x):
        return Likelihoods.gaussian_suff(x)

class MNISTGMM(nn.Module):
    
    def __init__(self, K, D=2, Dx = 2, nh=10):
        
        super().__init__()
        
        self.alpha = nn.Parameter(torch.ones(K), requires_grad=True)
        self.mu    = nn.Parameter(torch.randn(K, D)*0.5, requires_grad=True)
        #self.chol  = nn.Parameter(torch.stack([torch.eye(D,D)*0.3]*K), requires_grad=True)
        self.chol  = nn.Parameter(torch.ones(K, D)*0.3, requires_grad=False)
        
        #self.mean_network = Generators.dc_gen(nz=D, ngf=32, tanh=True)    
        self.mean_network = Generators.fc_gen(D, nh, Dx,tanh=True)    
        self.mean_network.apply(weights_init)
        self.log_std = nn.Parameter(torch.zeros(()), requires_grad=True)

        self.obs   = Likelihoods.GaussianObservation(self.mean_network, self.log_std)

        
    def sample_logp0(self, n):
        
        logp0 = 0
        
        #pz_pi= td.Categorical(self.alpha.softmax(0).clamp(1e-5,1-1e-5))
        pz_pi= td.Categorical(logits=self.alpha)
        z    = pz_pi.sample([n])
        logp0 = logp0 + pz_pi.log_prob(z)
        
        mean = self.mu[z,:]
        #cov  = torch.einsum("ijk,ilk->ijl", self.chol, self.chol)
        cov = self.chol
        #cov = cov + torch.eye(cov.shape[-1], device=cov.device)  * 1e-6
        cov = cov[z,...]
        
        py_z = td.Normal(mean, cov)
        y    = py_z.sample([])
        logp0 += py_z.log_prob(y).sum(-1)
                
        p   = self.obs.conditional_param(y)
        x    = p.sample([])
        nat  = self.obs.nat(y)
        norm = self.obs.norm(y)       
        x    = x.reshape(x.shape[0],-1)
        x    = x.reshape(y.shape[0],-1)
        
    
        return (None, z, None, None, y, p.mean.detach()), x, norm - logp0, nat

    
    def sample(self, n):
        return self.sample_logp0(n)[1]

    def suff(self, x):
        return Likelihoods.gaussian_suff(x)


class GP_GaussianLikelihood(nn.Module):
    
    def __init__(self, points, kernel, std, scale=1.0):
        super().__init__()
        self.kernel = kernel
        self.points = points
        self.device = points.device
        self.npoint = len(points)
        self.norm2  = self._compute_norm2(points, points)
        self.log_std = nn.Parameter(torch.tensor(math.log(std), device=self.device), requires_grad=True)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale), device=self.device), requires_grad=True)
        
    def sample(self, n):
        
        # have to add a diagonal, truncating singular values is unstable in pytorch!
        Sigma = self.gram() + torch.eye(self.npoint,device=self.device)*1e-4
        Z = td.MultivariateNormal(torch.zeros(self.npoint, device=self.device, dtype=Sigma.dtype), 
                                  covariance_matrix = Sigma)
        z = Z.sample([n])
        logp = Z.log_prob(z)
        
        X_Z = td.Normal(z, self.log_std.exp())
        x = X_Z.sample([])
        nat = Likelihoods.gaussian_nat(X_Z)
        norm= Likelihoods.gaussian_norm(X_Z)
        
        return z, x, nat, norm-logp
        
    def gram(self):
        
        return torch.exp(-0.5 * self.norm2 / self.kernel.log_sigma.exp()**2) * self.log_scale.exp()
        
    def _compute_norm2(self,X,Y):

#         K = (X**2).sum(-1, keepdim=True)
#         K = K + (Y**2).sum(-1, keepdim=True).t()
#         K -= 2 * (X[:,None,:] * Y[None,:,:]).sum(-1)
        
        K = ((X[:,None,:] - Y[None, :, :])**2).sum(-1)
        
        return K
    
    def cov(self, x, y):
        return self.kernel(x, y) * self.log_scale.exp()
    
    def posterior(self, y, X_new):
    
        KXX = self.cov(self.points, self.points)
        KxX = self.cov(X_new, self.points)

        KXX_new = self.cov(X_new, X_new)

        Sigma_inv = torch.inverse(KXX + torch.eye(y.shape[0], device=self.device) * self.log_std.exp()**2)

        m = KxX @ Sigma_inv @ y
        c = KXX_new - KxX @ Sigma_inv @ KxX.t()

        return m, c

    def condition(self, x, y, X_new):
    
        KXX = self.cov(x, x)
        KxX = self.cov(X_new, x)

        KXX_new = self.cov(X_new, X_new)

        Sigma_inv = torch.inverse(KXX + torch.eye(y.shape[0], device=self.device) * self.log_std.exp()**2)

        m = KxX @ Sigma_inv @ y
        c = KXX_new - KxX @ Sigma_inv @ KxX.t()

        return m, c



class HodgkinHuxley(nn.Module):
    
    def __init__(self, z0=None, C_m=None, g_l=None, E_l=None, g_na=None, E_na=None, 
                 g_k=None, E_k=None, V_t=None,
                 g_m=None, tau_max = None,
                 dt=1e-3, rand_std=0.5, noise_std=0.5):
        
        super().__init__()
         
        self.D    = 4
        self.z0   = nn.Parameter(torch.tensor(z0))
        self.C_m  = nn.Parameter(torch.tensor(math.log(C_m)))
        self.g_l  = nn.Parameter(torch.tensor(math.log(g_l)))
        self.E_l  = nn.Parameter(torch.tensor(math.log(-E_l)))
        self.g_na = nn.Parameter(torch.tensor(math.log(g_na)))
        self.E_na = nn.Parameter(torch.tensor(math.log(E_na)))
        self.g_k  = nn.Parameter(torch.tensor(math.log(g_k)))
        self.E_k  = nn.Parameter(torch.tensor(math.log(-E_k)))
        self.V_t  = nn.Parameter(torch.tensor(math.log(-V_t)))
#         self.g_m  = nn.Parameter(torch.tensor(math.log(g_m)))
#         self.tau_max = nn.Parameter(torch.tensor(math.log(tau_max)))
        self.noise_std = nn.Parameter(torch.tensor(math.log(noise_std)))
        self.rand_std = nn.Parameter(torch.tensor(math.log(rand_std)))

        self.dt   = dt

        
    def sample_logp(self, N, I, noise=None):
        
        T = len(I)
        g_l = self.g_l.exp()
        E_l = -self.E_l.exp()
        g_na= self.g_na.exp()
        E_na= self.E_na.exp()
        g_k = self.g_k.exp()
        E_k = -self.E_k.exp()
        dt  = self.dt
        C_m = self.C_m.exp()
        V_t = -self.V_t.exp()
#         g_m = self.g_m.exp()
#         tau_max = self.tau_max.exp()
        
        logp0 = 0
        z    = self.z0.expand(N, self.D)
        zs   = []
        Vs   = []
        
        # https://link.springer.com/content/pdf/10.1007%2Fs00422-008-0263-8.pdf
        
        for t in range(T):
            V = z[:,0]
            m = z[:,1]
            h = z[:,2]
            n = z[:,3]
#             p = z[:,4]
            
            # Na current
            a_m = -0.32 * (V - V_t - 13.) / (torch.exp(-(V - V_t - 13.) / 4.) - 1. )
            b_m =  0.28 * (V - V_t - 40.) / (torch.exp( (V - V_t - 40.) / 5.) - 1. )
            a_h =  0.128* 1              *  torch.exp(-(V - V_t - 17.) / 18.)
            b_h =  4.0  * 1              / (torch.exp(-(V - V_t - 40.) / 5.) + 1.)
            
            dm  = dt * (a_m * ( 1 - m ) - b_m * m)
            dh  = dt * (a_h * ( 1 - h ) - b_h * h)
            
            I_na= g_na * m**3. * h * (V - E_na)
            
            # K current
            
            a_n = -0.032 * (V - V_t - 15.) / (torch.exp(-(V - V_t - 15.) / 5.) - 1.)
            b_n = 0.5 * torch.exp( - ( V - V_t - 10.) / 40.)
            dn  = dt * (a_n * ( 1. - n ) - b_n * n)
            I_k = g_k * n**4 * (V - E_k)
            
            
#             # slow K current
#             p_inf = 1.0 / ( 1 + torch.exp(-( V + 35. ) / 10.))
#             tau_p = tau_max / (3.3 * torch.exp( (V + 35.) / 20.) + torch.exp( - (V + 35.) / 20.))
#             dp  = dt * (p_inf - p) / tau_p
#             I_m = g_m * p * (V - E_k)
            
#             dV  = dt / C_m * ( - g_l * ( V - E_l ) - I_na - I_k -I_m + I)
            dV  = dt / C_m * ( - g_l * ( V - E_l ) - I_na - I_k + I[t])
            
            V   = V + dV
            m   = m + dm
            h   = h + dh
            n   = n + dn
#             p   = p + dp

            dist = td.Normal(V, self.rand_std.exp())
            V = dist.sample()
            
            Vs += V,
            logp0 = logp0 + dist.log_prob(V)
#             z = torch.stack([V, m, h, n, p], -1).detach()
            z = torch.stack([V, m, h, n], -1)

            zs += z,
        
        zs = torch.stack(zs,1)[:,::1,:]
        Vs = torch.stack(Vs,1)[:,::1]
        
        if noise == None:
            noise_dist = td.Normal(Vs, self.noise_std.exp())
        else:
            noise_dist = td.Normal(Vs, max(noise, 1e-6))
        xs = noise_dist.sample()

        xs = xs.reshape(N, -1)
        nat = Likelihoods.gaussian_nat(noise_dist)
        norm = Likelihoods.gaussian_norm(noise_dist)
    
        return zs, xs, norm - logp0, nat

    
    def sample(self, n, I, noise=None):
        return self.sample_logp(n, I, noise)[:2]
    
    def suff(self, x):
        return Likelihoods.gaussian_suff(x)
    
class Ecology(nn.Module):
    
    def __init__(self, z_0=0.0, P=None, N_0=None, s_d=None, s_p=None, tau_count=25, delta=None, 
                 noise_std=0.5, rand_std=1.0,burnin=0, shift=0.0, fit_all_z_0=True):
        
        super().__init__()
         
        self.D    = 1
        self.P    = nn.Parameter(torch.tensor(math.log(P)))
        self.N_0  = nn.Parameter(torch.tensor(math.log(N_0)))
        
        if fit_all_z_0:
            self.z_0  = nn.Parameter(torch.tensor([z_0] * tau_count))
        else:
            self.z_0  = nn.Parameter(torch.tensor(z_0))
        self.s_d  = nn.Parameter(torch.tensor(math.log(s_d)))
        self.s_p  = nn.Parameter(torch.tensor(math.log(s_p)))
        self.tau_logits = nn.Parameter(torch.zeros(tau_count))
        self.delta = nn.Parameter(torch.tensor(math.log(delta)))
        self.noise_std = nn.Parameter(torch.tensor(math.log(noise_std)))
        self.rand_std = nn.Parameter(torch.tensor(math.log(rand_std)), requires_grad=True)
        self.burnin = burnin
        self.shift  = shift
        self.tau_count = tau_count
        self.fit_all_z_0 = fit_all_z_0

    def sample_logp(self, N, I, noise=None):
        
        max_tau = self.tau_count
        burnin = self.burnin 
        logp0  = 0.0
        
        T = len(I)
        P     = self.P.exp()
        N_0   = self.N_0.exp()
        s_d   = self.s_d.exp()
        s_p   = self.s_p.exp()
        delta = self.delta.exp()
        
        p_tau = td.Categorical(logits=self.tau_logits)
        tau   = p_tau.sample([N])
        logp0 = logp0 + p_tau.log_prob(tau)

        tau   = tau + 1
            
        #tau = self.tau_logits.softmax(0)
        
        p_et = td.Gamma(1 / s_p**2, 1 / s_p**2)
        et = p_et.sample([N, T+burnin])
        logp0 = logp0 + p_et.log_prob(et).sum(1)

        p_epsilont = td.Gamma(1 / s_d**2, 1 / s_d**2)
        epsilont = p_epsilont.sample([N, T+burnin])
        logp0 = logp0 + p_epsilont.log_prob(epsilont).sum(-1)
        
                
        #z = self.z_0.exp().expand(N)
        zs = torch.zeros(N, max_tau+burnin+T, dtype=P.dtype, device=P.device)
        zs[torch.arange(N),:max_tau] = self.z_0.sigmoid()
        
        z = zs[torch.arange(N),max_tau-1]    
        for t in torch.arange(max_tau, T+burnin+max_tau):
        
            ztmtau = zs[torch.arange(N), t-tau]
            # print("----")
            # print(zs[:,:t+1])
            # print(tau_t)
            # print(ztmtau)
            # tstart = max(0, t+1-len(tau))
            # ztmtau = zs[torch.arange(N), tstart:t+1] @ tau[:min(t,len(tau))+1]
            z = P * ztmtau * (-ztmtau / N_0).exp() * et[:,t-max_tau] + z * (-delta * epsilont[:,t-max_tau]).exp()
            
            pz = td.LogNormal(z.log(), self.rand_std.exp())
            z  = pz.sample([])
            logp0   = logp0 + pz.log_prob(z)
            zs[:,t] = z
        
        xs = zs[:, -T:].detach()
        
        '''
        if noise is None:
            noise_dist = td.Normal(xs, self.noise_std.exp())
        else:
            noise_dist = td.Normal(xs, max(noise, 1e-6))
        xs = noise_dist.sample([])
        xs = xs.reshape(N, -1)
        nat = Likelihoods.gaussian_nat(noise_dist)
        norm = Likelihoods.gaussian_norm(noise_dist)
        return (tau, et, epsilont, zs), xs, norm - logp0, nat
        '''

        return (tau, et, epsilont, zs), xs, -logp0, torch.zeros(N,T*2,dtype=xs.dtype,device=xs.device)
    
    def sample(self, n, I, noise=None):
        return self.sample_logp(n, I, noise)[:2]
    
    def suff(self, x):
        return Likelihoods.gaussian_suff(x)
