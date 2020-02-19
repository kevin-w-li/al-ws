import torch, math
from torch import nn, optim
from scipy.spatial.distance import pdist
import numpy as np
import torch.distributions as td
import pdb 

def estimate_median_distance(data):
    return np.median(pdist(data.detach().cpu().numpy()))

def l2_distance(FX,FY, stable):

    if stable:
        FK = torch.sum((FX[:,None,:] - FY[None,:,:])**2,-1)
    else:
        FK = (FX**2).sum(-1, keepdim=True)
        FK = FK + (FY**2).sum(-1, keepdim=True).t()
        FK -= 2 * (FX[:,None,:] * FY[None,:,:]).sum(-1)
    
    return FK


class PolynomialKernel(nn.Module):

    def __init__(self, nets=[torch.nn.Identity()], sigma=1.0, c=1.0,  p=2.0, lam=0.01, train_lam=False, train_sigma=False):

        super().__init__() 

        Fdata = []
        self.n = len(nets)
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam)), requires_grad=train_lam)
        self.c = c
        self.p = p
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)), requires_grad=train_sigma)
        self.kernel_networks = nn.ModuleList(nets)

    def forward(self, x, y):
        return self.gram(x, y)

    def dgg(self, X,Y,g):
        
        # embedding of X
        FX = g(X)
        FY = g(Y)

        return FX@FY.T

    def gram(self, X, Y):
        # compute the gram matrix 
        nx, sx = X.shape
        ny, sy = Y.shape
        FK = 0.0
        for k in self.kernel_networks:
            FK = FK + self.dgg(X, Y, k)

        G = (self.c + FK / self.log_sigma.exp())** self.p
        return G
    
    
    
class SetGaussianKernel(nn.Module):

    '''
    set kernel as introduced in https://arxiv.org/pdf/1910.04086.pdf
    there is a mistake in equation (3) where the first and third term should be double sums 
    from the respective two sets S and S'
    '''

    def __init__(self, nets=[torch.nn.Identity()], sigma=1.0,  theta_out=1.0, lam=0.01, train_sigma=False, train_lam=False, stable=False, reweight=False):

        super().__init__() 

        Fdata = []
        self.n = len(nets)
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam)), requires_grad=train_lam)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)), requires_grad=train_sigma)
        self.log_theta_out = nn.Parameter(torch.tensor(math.log(theta_out)), requires_grad=train_sigma)
        self.kernel_networks = nn.ModuleList(nets)
        self.stable = stable 
        self.reweight = reweight

    def forward(self, x, y):
        return self.gram(x, y)

    def dgg(self, X,Y,g):
        
        # embedding of X
        FX = g(X)
        FY = g(Y)
        XX =  ((FX[:,:,None] - FX[:,None,:])**2)
        YY =  ((FY[:,:,None] - FY[:,None,:])**2)
        XY =  ((FX[:,None,:,None] - FY[None,:,None,:])**2)
        return XX, YY, XY

    def gram(self, X, Y):
        # compute the gram matrix 
        nx, sx = X.shape
        ny, sy = Y.shape
        XY = 0 
        XX = 0 
        YY = 0 
        for k in self.kernel_networks:
            xx, yy, xy = self.dgg(X, Y, k)
            XY = XY + xy
            XX = XX + xx
            YY = YY + yy

        KXX = torch.exp(- 0.5 * XX / (self.log_sigma.exp()**2)).mean([1,2])
        KYY = torch.exp(- 0.5 * YY / (self.log_sigma.exp()**2)).mean([1,2])
        KXY = torch.exp(- 0.5 * XY / (self.log_sigma.exp()**2)).mean([2,3])

        FK = ( KXX[:,None] * KYY[None,:] - 2. * KXY)

        G =  (-0.5*FK/(self.log_theta_out.exp()**2)).exp()

        return G
    


class RFKernel(nn.Module):
    
    def __init__(self, nfeat_in, nets=[torch.nn.Identity()], p=math.inf, sigma=1.0, lam = 0.001, nfeat = 300, train_sigma=False, train_lam=False, stable=False, reweight=False):

        super().__init__()
        
        Fdata= []
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam)), requires_grad=train_lam)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)), requires_grad=train_sigma)
        self.kernel_networks = nets
        self.nfeat_in = nfeat_in
        self.nfeat = nfeat
        self.p = p
        if self.p == math.inf:
            self.W = nn.Parameter(torch.randn(nfeat_in, nfeat), requires_grad=False)
        elif self.p == 0.5:
            self.W = nn.Parameter(td.Cauchy(0,1).sample([nfeat_in, nfeat]), requires_grad=False)
        else:
            raise NotImplementedError("wrong p")
        self.b = nn.Parameter(torch.rand(nfeat) * 2 * math.pi, requires_grad=False)
        

    def forward(self,x,y):
        return self.gram(x, y)

    def feat(self, X):
        
        X = torch.cat([n(X) for n in self.kernel_networks],0)
        D = X.shape[-1]
        if self.p == math.inf:
            return torch.cos( X @ self.W / self.log_sigma.exp()  + self.b )
        elif self.p == 0.5:
            return torch.cos( X @ self.W / self.log_sigma.exp()  + self.b )

    def gram(self, X, Y):
        X = self.feat(X)
        Y = self.feat(Y)
        return X@Y.T

    def regenerate(self):
        if self.p == math.inf:
            self.W.data = torch.randn(self.nfeat_in, self.nfeat, device=self.W.device, dtype=self.W.dtype)
        elif self.p == 0.5:
            self.W.data = nn.Parameter(td.Cauchy(0,1).sample([nfeat_in, nfeat], device=self.W.device, dtype=self.W.dtype))
        else:
            raise(NotImplementedError, "wrong p")
        self.b.data = torch.rand(self.nfeat, device=self.W.device, dtype=self.W.dtype) * 2 * math.pi

class Kernel(nn.Module):
    
    def __init__(self, nets=[torch.nn.Identity()], sigma=1.0, lam = 0.001, p=math.inf, train_sigma=False, train_lam=False, stable=False, reweight=False):

        super().__init__()
        
        Fdata= []
        self.n = len(nets)
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam)), requires_grad=train_lam)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)), requires_grad=train_sigma)
        self.kernel_networks = nn.ModuleList(nets)
        self.stable = stable
        self.reweight = reweight
        self.p = p

    def forward(self,x,y):
        return self.gram(x, y)

    def mmd(self, X, Y):
        
        G1 = self.gram(X, X).mean()
        G2 = self.gram(Y, Y).mean()
        G3 = self.gram(X, Y).mean()
        return G1 + G2 - 2 * G3

    def dgg(self, X,Y,g):
        
        FX = g(X)
        if FX.shape[-1] == 1 and len(FX.shape) == 3:
            h, w, _ = FX.shape 
            FX = FX.view(h, w)
        FY = g(Y)
        if FY.shape[-1] == 1 and len(FY.shape) == 3:
            h, w, _ = FY.shape 
            FY = FY.view(h, w)
        
        FK = l2_distance(FX, FY, self.stable)

        return FK

    def kmm_weight(self, X,Y):

        ''' importance weight to correct for covariate shift, from Kernel Mean Matching (Gretton, 2003)
        '''
        
        X = X
        Y = Y
        
        n1 = X.shape[0]
        n2 = Y.shape[0]
        w = np.ones(n1) / n1
        
        import  cvxpy as cp

        K = self.gram(X, X).detach().cpu().numpy()
        K = K + np.eye(n1)*1e-9
        k = self.gram(X, Y).detach().cpu().numpy().sum(-1) * n1 / n2
        
        ones = np.ones(n1)
        zeros = np.zeros(n1)
        eyes = np.eye(n1)
        x = cp.Variable(n1)
        prob = cp.Problem(cp.Minimize( (0.5) * cp.quad_form(x, K) - k.T @ x), 
                [ -x <= zeros, x<=10, ones @ x==n1])
        prob.solve()

        x = torch.tensor(x.value, device=X.device, dtype = X.dtype)
        
        return x

    def gram(self, X, Y):

        G = 0
        for k in self.kernel_networks:
            G = G + self.dgg(X, Y, k)

        if self.p == math.inf:
            G =  (-G/(2*self.log_sigma.exp()**2)).exp()
            return G
        else:
            #G =  (-G/(self.log_sigma.exp())).exp()
            distance = G.sqrt() / self.log_sigma.exp()
            exp_component = torch.exp(-math.sqrt(self.p * 2) * distance)

            if self.p == 0.5:
                constant_component = 1
            elif self.p == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.p == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component



    def update_normlisation(self, X):
        for k in self.kernel_networks:
            k.train()
            k(X)
            k.eval()
        return None

class SetKernel(nn.Module):

    def __init__(self, nets=[torch.nn.Identity()], sigma=1.0,  lam=0.01, train_sigma=False, train_lam=False, stable=False, reweight=False):
        """
        implement a kernel over sets
        at the moment, only supports a linear input kernel and Gaussian output kernel 
        """
        super().__init__() 

        Fdata = []
        self.n = len(nets)
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam)), requires_grad=train_lam)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)), requires_grad=train_sigma)
        self.kernel_networks = nn.ModuleList(nets)
        self.stable = stable 
        self.reweight = reweight

    def forward(self, x, y):
        return self.gram(x, y)

    def dgg(self, X,Y,g):
        
        # embedding of X
        FX = g(X)
        if FX.shape[-1] == 1 and len(FX.shape) == 3:
            n, sx, _ = FX.shape
            FX = FX.view(n, sx)
        nx, sx = FX.shape
        # embedding of Y
        FY = g(Y)
        if FY.shape[-1] == 1 and len(FY.shape) == 3:
            n, sy, _ = FY.shape 
            FY = FY.view(n, sy)
        ny, sy = FY.shape

        # simulated set-wise operations 
        XX = torch.bmm(FX[:,:,None], FX[:,None,:]).sum([1, 2]) / (sx ** 2)
        YY = torch.bmm(FY[:,:,None], FY[:,None,:]).sum([1, 2]) / (sy ** 2)
        XY = torch.einsum('ns,mt->nmst', FX, FY).sum([2, 3]) / (sx * sy)
        FK = XX.view(nx, 1) + YY.view(1, ny) - 2 * XY

        return FK

    def gram(self, X, Y):
        # compute the gram matrix 
        G = 0 
        for k in self.kernel_networks:
            G = G + self.dgg(X, Y, k)
        G =  (-G/(2*self.log_sigma.exp()**2)).exp()

        return G


def draw_sample(prior, obs, n, detach=True):
    with torch.no_grad():
        z = prior.sample([n])
        x = obs.sample(z)
    return x

def draw_mean(prior, obs, n):
    with torch.no_grad():
        z = prior.sample([n])
        x = obs.conditional_param(z)
    if isinstance(x, torch.distributions.Distribution):
        x = x.mean
    return x.reshape(x.shape[0],-1)

def draw_joint_sample(prior, obs, n, detach=True):
    z = prior.sample([n])
    x = obs.sample(z, detach=detach)
    return z, x

def sleep(prior, obs, nsleep, detach=False, detach_sample=True, train_prior=False):
    
    sleep_z, sleep_x = draw_joint_sample(prior, obs, nsleep, detach=detach_sample)
    
    if detach:
        with torch.no_grad(): 
            norm  = obs.norm(sleep_z)
            nat   = obs.nat(sleep_z)
            if train_prior:
                logp0 = prior.log_prob(sleep_z).sum(-1)
    else:
        norm  = obs.norm(sleep_z)
        nat   = obs.nat(sleep_z)
        if train_prior:
            logp0 = prior.log_prob(sleep_z).sum(-1)
        
    if train_prior:
        norm = norm - logp0
    
    return sleep_z, sleep_x, norm, nat

def flow_sleep(prior, obs, nsleep):

    z, logp0 = prior.sample_logp([nsleep])
    norm = obs.norm(z)
    nat  = obs.nat(z)
    x    = obs.sample(z)
    return z, x, norm - logp0, nat

def weights_init(m, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, std)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, std)
        m.bias.data.fill_(0)

def set_weights_init(std):
    def weights_init(m, std=std):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, std)
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, std)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, std)
            m.bias.data.fill_(0)
    return weights_init
     

def save_kws(fn, model, kernel=None, optimizer=None, kernel_optimizer=None):
    dicts = {}
    dicts["model"] = model.state_dict()
    if kernel is not None:
        dicts["kernel"] = kernel.state_dict()
    if optimizer is not None:
        dicts["optimizer"] = optimizer.state_dict()
    if kernel_optimizer is not None:
        dicts["kernel_optimizer"] = optimizer.state_dict()

    torch.save(dicts, 
               fn)

def load_kws(fn, model, kernel=None, optimizer=None, kernel_optimizer=None):
    
    d = torch.load(fn)
    model.load_state_dict(d["model"])
    if kernel is not None:
        kernel.load_state_dict(d["kernel"])
    if optimizer is not None:
        optimizer.load_state_dict(d["optimizer"])

    if kernel_optimizer is not None:
        kernel_optimizer.load_state_dict(d["kernel_optimizer"])
    
def KRR_old(suff, deep_gaussian_gram, wake_x, sleep_x, norm, nat, log_sigma, log_lam, noise = 0.0):

    # depreciated version of regression, do not use

    nsleep = sleep_x.shape[0] 
    S_tr = suff(wake_x)
    K = (deep_gaussian_gram(sleep_x, sleep_x, log_sigma.exp()) + 
         torch.eye(nsleep, device=sleep_x.device, dtype=sleep_x.dtype) * log_lam.exp())
    
    G = deep_gaussian_gram(sleep_x, wake_x, log_sigma.exp())

    GKinv = torch.solve(G, K)[0].t()
    Elogp = torch.mean( GKinv @ ( - norm) + 
                         (( GKinv @ nat ) * S_tr).sum(-1)
                       )
    return Elogp

def KRR_ind(suff, deep_gaussian_gram, wake_x, sleep_x, norm, nat, log_sigma, log_lam, noise = 0.0):

    # depreciated version of regression, do not use

    nsleep = sleep_x.shape[0] 
    S_tr = suff(wake_x)
    K = (deep_gaussian_gram(sleep_x, sleep_x, log_sigma.exp()) + 
         torch.eye(nsleep, device=sleep_x.device, dtype=sleep_x.dtype) * log_lam.exp())
    
    G = deep_gaussian_gram(sleep_x, wake_x, log_sigma.exp())

    GKinv = torch.solve(G, K)[0].t()
    Elogp = GKinv @ ( - norm) + (( GKinv @ nat ) * S_tr).sum(-1)

    return Elogp

def KRR_rf(kernel, wake_x, sleep_x, *fs, reweight=None):
    
    '''
    Kerner ridge regression that takes sleep data [`sleep_x`] and potentailly more than 1 functiosn `f`
    and produces predictions given wake data `wake_x`

    kernel:     the kernel used for KRR
    wake_x:     real data, size: nwake by Dx
    sleep_x:    simulated data from generative model, size: nsleep by Dx
    fs:         sleep target functions evaluated at latent (sleep_z), size: nsleep (by f.shape[1:])
    
    The formula for KRR prediction given one target `f` from `fs` is:  
                z_pred = k(wake_x, sleep_x) @ k(sleep_x, sleep_x)^{-1} @ f
                       = G @ K^{-1} @ f

    '''

    nsleep = sleep_x.shape[0] 
    for fi, f in enumerate(fs):
        assert f.shape[0] == nsleep, "%1d'th target has wrong number of nsleep"


    # regressions share the same inverse of the gram matrix K(sleep_x, sleep_x)
    # K = (kernel(sleep_x, sleep_x) + torch.diag(1.0/w) * kernel.log_lam.exp())

    sleep_phi = kernel.feat(sleep_x)
    wake_phi  = kernel.feat(wake_x)
    w = torch.eye(sleep_phi.shape[-1], device=sleep_x.device, dtype=sleep_x.dtype) 
    K = sleep_phi.T @ sleep_phi + torch.diag(1.0/w) * kernel.log_lam.exp()

    GKinv = wake_phi @ torch.solve(sleep_phi.T, K)[0]
    
    return (GKinv @ f for f in fs)


def KRR(kernel, wake_x, sleep_x, *fs, reweight=None):
    
    '''
    Kerner ridge regression that takes sleep data [`sleep_x`] and potentailly more than 1 functiosn `f`
    and produces predictions given wake data `wake_x`

    kernel:     the kernel used for KRR
    wake_x:     real data, size: nwake by Dx
    sleep_x:    simulated data from generative model, size: nsleep by Dx
    fs:         sleep target functions evaluated at latent (sleep_z), size: nsleep (by f.shape[1:])
    
    The formula for KRR prediction given one target `f` from `fs` is:  
                z_pred = k(wake_x, sleep_x) @ k(sleep_x, sleep_x)^{-1} @ f
                       = G @ K^{-1} @ f

    '''

    nsleep = sleep_x.shape[0] 
    for fi, f in enumerate(fs):
        assert f.shape[0] == nsleep, "%1d'th target has wrong number of nsleep"
    
    if hasattr(kernel, "reweight") and reweight is None:
        reweight = kernel.reweight

    if isinstance(reweight, torch.Tensor):
        m = torch.diag(1.0/reweight)

    elif reweight:
        w = kernel.kmm_weight(sleep_x, wake_x)
        m = torch.diag(1.0/w)
    else:
        m = torch.eye(nsleep, device=sleep_x.device)

    # regressions share the same inverse of the gram matrix K(sleep_x, sleep_x)
    K = kernel(sleep_x, sleep_x) + m * kernel.log_lam.exp()
        
    # similarity between sleep and wake data
    G = kernel(sleep_x, wake_x)
    
    # this computes G @ K^{-1}
    n = K.shape[0] # + 0.0001 * torch.eye(n, device=sleep_x.device)
    GKinv = torch.solve(G, K)[0].t()

    return (GKinv @ f for f in fs)


def KRR_2(suff, kernel, wake_x, sleep_x, norm, nat, reweight=None, rf=False):
    
    '''
    Kerner ridge regression that takes sleep data [`sleep_x`, `norm`, `nat`]
    and produces predictions given wake data `wake_x`

    suff:       sufficient statistics of the likelihood
    kernel:     the kernel used for KRR
    wake_x:     real data, size: nwake by Dx
    sleep_x:    simulated data from generative model, size: nsleep by Dx
    norm:       log normalizer of likelihood evaluated at latent (sleep_z), size: nsleep
    nat:        natural parameters of likelihood evaluated at latent (sleep_z), size: nsleep x Dsuff
    
    The formula for KRR prediction is:  
                z_pred = k(wake_x, sleep_x) @ k(sleep_x, sleep_x)^{-1} @ f
                       = G @ K^{-1} @ f
     where f_sleep can be `norm` or `nat`
    
    In Algorithm 1 of the paper:
    h^\Psi(x) = k( x , sleep_x) @ k(sleep_x, sleep_x)^{-1} @ norm
    h^\eta(x) = k( x , sleep_x) @ k(sleep_x, sleep_x)^{-1} @ nat

    '''
    S_tr = suff(wake_x)
    if len(S_tr.shape) == 3:
        S_tr = S_tr.view(nat.shape)
    
    # use KRR to perform KRR training and evaluation for norm and nat
    if isinstance(kernel, RFKernel):
        Enat, Enorm = KRR_rf(kernel, wake_x, sleep_x, nat, norm, reweight=reweight)
    else:
        Enat, Enorm = KRR(kernel, wake_x, sleep_x, nat, norm, reweight=reweight)

    # this computes \hat{J_\theta} in the paper
    Elogp = (Enat * S_tr).sum(-1) - Enorm
    return Elogp

def KRR_3(suff, kernel, wake_x, sleep_x, norm, nat, sleep_z):
    
    '''
    Kerner ridge regression that takes sleep data [`sleep_x`, `norm`, `nat`]
    and produces predictions given wake data `wake_x`

    suff:       sufficient statistics of the likelihood
    kernel:     the kernel used for KRR
    wake_x:     real data, size: nwake by Dx
    sleep_x:    simulated data from generative model, size: nsleep by Dx
    norm:       log normalizer of likelihood evaluated at latent (sleep_z), size: nsleep
    nat:        natural parameters of likelihood evaluated at latent (sleep_z), size: nsleep x Dsuff
    
    The formula for KRR prediction is:  
                z_pred = k(wake_x, sleep_x) @ k(sleep_x, sleep_x)^{-1} @ f
                       = G @ K^{-1} @ f
     where f_sleep can be `norm` or `nat`
    
    In Algorithm 1 of the paper:
    h^\Psi(x) = k( x , sleep_x) @ k(sleep_x, sleep_x)^{-1} @ norm
    h^\eta(x) = k( x , sleep_x) @ k(sleep_x, sleep_x)^{-1} @ nat

    '''
    S_tr = suff(wake_x)
    if len(S_tr.shape) == 3:
        S_tr = S_tr.view(nat.shape)
    
    # use KRR to perform KRR training and evaluation for norm and nat
    Enat, Enorm, Ez= KRR(kernel, wake_x, sleep_x, nat, norm, sleep_z)

    # this computes \hat{J_\theta} in the paper
    Elogp = (Enat * S_tr).sum(-1) - Enorm
    return Elogp, Ez


def cKRR(obs, deep_gaussian_gram, wake_x, sleep_x, norm, nat, log_sigma, log_lam, noise=0.0, bs = 1000):

    '''
    intended for chunking the gram matrix on GPU and compute inverse on CPU
    '''

    nsleep = sleep_x.shape[0] 
    bs = min(nsleep, bs)
        
    S_tr = suff(wake_x)
    if noise != 0:
        sleep_x = sleep_x + torch.randn_like(sleep_x)*noise
        
    nb = nsleep//bs
    K  = torch.zeros(nsleep, nsleep)
    for i in range(nb):
        for j in range(i,nb):
            si = slice(i*bs,(i+1)*bs)
            sj = slice(j*bs,(j+1)*bs)
            di = sleep_x[si]
            dj = sleep_x[sj]
            K[si, sj] = deep_gaussian_gram(di, dj, log_sigma.exp())
            if i!=j:
                K[sj, si] = K[si, sj].t()
            
    K += torch.eye(nsleep) * log_lam.exp()
    
    if noise != 0: 
        sleep_x = sleep_x + torch.randn_like(sleep_x)*noise
    G = deep_gaussian_gram(sleep_x, wake_x, log_sigma.exp()).cpu()


    GKinv = torch.solve(G, K)[0].t()
    Elogp = torch.mean( GKinv @ ( - norm.cpu()) + 
                         (( GKinv @ nat.cpu() ) * S_tr.cpu()).sum(-1)
                       )
    return Elogp.to(sleep_x.device)

def NKRR(kernel, wake_x, sleep_x, npoint, *fs):
    
    '''
    Nystrom kerner ridge regression that takes sleep data [`sleep_x`] and potentailly more than 1 functiosn `f`
    and produces predictions given wake data `wake_x`

    kernel:     the kernel used for KRR
    wake_x:     real data, size: nwake by Dx
    sleep_x:    simulated data from generative model, size: nsleep by Dx
    fs:         sleep target functions evaluated at latent (sleep_z), size: nsleep (by f.shape[1:])
    
    The formula for KRR prediction given one target `f` from `fs` is:  
                z_pred = k(wake_x, sleep_x) @ k(sleep_x, sleep_x)^{-1} @ f
                       = G @ K^{-1} @ f

    '''

    nsleep = sleep_x.shape[0] 
    for fi, f in enumerate(fs):
        assert f.shape[0] == nsleep, "%1d'th target has wrong number of nsleep"
    
    idx = np.random.choice(nsleep, npoint, replace=False)
    points = sleep_x[idx]

    Kmm = kernel(points, points)
    Knm = kernel(sleep_x, points)
        
    G   = kernel(wake_x, points)

    GKinv = G @ torch.solve(Knm.t(), Knm.t() @ Knm + torch.eye(npoint, device=G.device)* kernel.log_lam.exp())[0]

    return (GKinv @ f for f in fs)


def NKRR_2(suff, kernel, wake_x, sleep_x, norm, nat, noise=0.0, npoint = 300):

    '''
    Nystrom KRR where a subset of sleep samples are used as inducing points
    '''

    S_tr = suff(wake_x)
    if len(S_tr.shape) == 3:
        S_tr = S_tr.view(nat.shape)
    
    # use KRR to perform KRR training and evaluation for norm and nat
    Enat, Enorm = NKRR(kernel, wake_x, sleep_x, npoint, nat, norm)

    # this computes \hat{J_\theta} in the paper
    Elogp = (Enat * S_tr).sum(-1) - Enorm
    return Elogp

def to_np(*args):
    return (a.cpu().detach().numpy() if type(a) is not np.ndarray else a for a in args)
