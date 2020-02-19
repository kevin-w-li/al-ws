import numpy as np
import torch
import tqdm
from NeuralProcess  import NeuralProcess, SineData, context_target_split
from torch import optim as opt
from KernelWakeSleep import Kernel, KRR_2, save_kws
import Likelihoods

device=torch.device("cuda:0")
# Create dataset
seed = 1
nepoch = 50
np.random.seed(seed)
dataset = SineData(amplitude_range=(-0.5, 1.),
                   shift_range=(-0.5,0.5),
                   num_samples=10000, seed=seed, noise_std =0.1)

batch_size = 1
num_context=4
n_samples=5000
num_target =4


data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

kernel = Kernel(sigma=1.0, lam=1e-3, train_lam=True, train_sigma=True, stable=True).to(device)


fn = "sin_ns%d_ne%d_s%02d"%(n_samples, nepoch, seed)

x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 50  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

NP = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim).to(device)


optimizer = opt.Adam(NP.model_parameters, lr=1e-4) # 1e-3 is unstable, 3e-4 also works
kernel_optimizer = opt.Adam(kernel.parameters(), lr=1e-4)


for ei in range(nepoch):
    for di, data in enumerate(tqdm.tqdm((data_loader))):

        # generate data
        x, y = data
        x = x.to(device)
        y = y.to(device)
        x_context, y_context, x_target, y_target = context_target_split(x, y, num_context, num_target) 
        # assume that there data dimension is 1
        x_context = x_context[0]
        y_context = y_context[0]
        y_target = y_target[0]
        x_target = x_target[0]

        # compute q_context
        q_context = NP.context_to_q(x_context, y_context)

        if di % 1 == 0:
            # optionally train kernel with cross validation
            y_sample, nat, norm = NP.q_to_samples(q_context
                                               , x_target, n_samples)
            y_sample_val, nat_val, norm_val = NP.q_to_samples(q_context, x_target, 100)
            Elogp_val = KRR_2(Likelihoods.gaussian_suff, kernel, y_sample_val, y_sample, norm, nat)  
            Elogp_true = (Likelihoods.gaussian_suff(y_sample_val) * nat_val).sum(-1) - norm_val
            loss = torch.mean((Elogp_val - Elogp_true)**2)
            kernel_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            kernel_optimizer.step()

        
        # update NP by amortized learning
        y_sample, nat, norm = NP.q_to_samples(q_context, x_target, n_samples)
        Elogp = KRR_2(Likelihoods.gaussian_suff, kernel, y_target.T, y_sample, norm, nat).mean()
        loss = -Elogp

#         # original 
        
#         # compute q_target
#         q_target = context_to_q(x_target, y_target)
#         kl = torch.distributions.kl.kl_divergence(q_context, q_target).sum()
#         y_pred   = q_to_y_pred(q_target, x_target)
#         Elogp = y_pred.log_prob(y_target).sum(-1).sum()
#         loss = -Elogp + kl
        
        #Elogps += Elogp.detach().cpu().numpy(),

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if di% 100 == 99:
            save_kws("models/kws/NPs/amortised/%s_ep%02d" % (fn, ei), NP, kernel, optimizer, kernel_optimizer)
