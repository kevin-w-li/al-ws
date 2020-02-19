import os
os.environ["MKL_NUM_THREADS"]="8" 
os.environ["NUMEXPR_NUM_THREADS"]='8'
os.environ["OMP_NUM_THREADS"]='8'

import matplotlib.pyplot as plt
import types
import numpy as np
import torch, tqdm
from torch import optim
from sklearn.decomposition import FastICA
import torchvision
import Generators, Features, Data, Likelihoods
import KernelWakeSleep as kws
from Models import ICA


# set a few parameters
device = torch.device("cuda:0")
batch_size = 100
dname = "van"
# number of channels
nc = 1                  
D0 = 36
nsleep = 2000           
nepoch = 100
# validation set for training gradient model parameters
nvalid = 200            
image_size = 16
# number of features used for the linear projection
nfeat = 300

# training data size
ntrain = 10**5          
# testing data size, not used here
ntest = 10**4
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

fn = "ica_D%d_nf%d_s%02d" % (D0, nfeat, seed)

print(fn)

# compute ICA features
_, natural_images = Data.load_data(dname, ntrain=ntrain, ntest=ntest, image_size=image_size, seed=seed,
                                           batch_size=batch_size)

Dx = natural_images.shape[-1]
ica = FastICA(n_components=D0, max_iter=1000)
ica.fit(natural_images)
A = torch.tensor(ica.mixing_, dtype=torch.float, device=device)

# normalise filteres
A -= A.mean(dim=0)
A /= A.norm(dim=0)

# generative model
model = ICA(D0, Dx, device).to(device)

# set training image from true A
train_images = model.prior.sample([ntrain]) @ A.to(device).T 
#test_images  = model.prior.sample([ntest])  @ A.to(device).T

def pre_process(data):
    data = data - data.mean(0, keepdims=True)
    clear_data = data / data.std(0, keepdims=True)
    data = clear_data + torch.randn(*clear_data.shape, device=device) * 0.1

    return clear_data, data
clear_train_images, train_images = pre_process(train_images)
#clear_test_images, test_images = pre_process(test_images)

data = model.sample(nsleep)
# compute paramters of gradient model
block_size=100
kernel_networks = []
Fdata = []
nnet = nfeat // block_size
if nnet > 0:
    for i in range(nnet):

        k = Features.lin_feat(Dx, block_size, bn=False).to(device)
        k.apply(kws.weights_init)
        Fdata += k(data),
        kernel_networks += k,

else:
    k = torch.nn.Identity().to(device)
    Fdata += k(data),
    kernel_networks += k,
    
Fdata   = torch.cat(Fdata, -1) 
Fmedian =  kws.estimate_median_distance(Fdata)
kernel = kws.Kernel(kernel_networks, Fmedian, lam=1e-3).to(device)



kwargs = {'num_workers': 4, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(train_images)),
        batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
kernel_optimizer = optim.Adam(kernel.parameters(), lr=0.01)


Elogps = []
losses = []


nsleep = 2000

model.train()
kernel.train()

for ei in tqdm.tqdm(range(100), ncols=100, unit="epoch"):
    
    with tqdm.tqdm(train_loader, leave=False, ncols = 100, unit="iter") as t:
        for i, (X_tr,) in enumerate(t):
        
            X_tr = X_tr.float().to(device)

            data_0, data_1, nat, norm = model.sample_logp(nsleep)
            data_0_val, data_1_val, nat_val, norm_val = model.sample_logp(nsleep)
            S_val = model.suff(data_1_val)
            
            # kernel update
            kernel_optimizer.zero_grad()
            Elogp_val = kws.KRR_2(model.suff, kernel, data_1_val, data_1, norm.detach(), nat.detach())
            logp_true= (nat_val * S_val).sum(-1) - norm_val 
            loss_val = torch.mean((Elogp_val -logp_true)**2)
            loss_val.backward(retain_graph=False)
            kernel_optimizer.step()
            losses += loss_val.detach().cpu().numpy(),
            
            # regression
            data_0, data_1, nat, norm = model.sample_logp(nsleep)
            Elogp  = kws.KRR_2(model.suff, kernel, X_tr, data_1, norm, nat).mean()   
            gen_loss = -Elogp
            optimizer.zero_grad()
            gen_loss.backward(retain_graph=False)
            optimizer.step()

            Elogps += Elogp.detach().cpu().numpy(),
            t.set_postfix(Elogp=Elogp.item(), std=model.log_noise.exp().item(), sigma=kernel.log_sigma.exp().item())
                        

    torch.save([A, model.state_dict(), kernel.state_dict(), optimizer.state_dict(), kernel_optimizer.state_dict()], "models/kws/%s" % fn)
    A_est = model.gen_network[1].weight.cpu().detach().numpy()
    np.savez("results/kws/%s_ep%02d.npz" % (fn, ei), A_est)
