import numpy as np
import torch
import torch.utils.data

def load_data(dname, binary=False, image_size=32, ntrain=5*10**4, ntest=10**4, seed=0, batch_size = 100, tanh=False, return_loader=True):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dname == "van":

        import vanhateren

        vh = vanhateren.VanHateren(calibrated=True)
        patches = vh.patches(ntrain+ntest, (image_size,image_size)).reshape(ntrain+ntest,-1)
        patches = vanhateren.preprocess.scale(patches).astype("float32")

        train_images = patches[:ntrain]
        test_images  = patches[ntrain:]
        print(np.mean(np.linalg.norm(test_images,axis=-1)))
        Dx = train_images.shape[-1]

        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_images)),
            batch_size=batch_size, shuffle=True)

    elif dname == "fmnist":
        import h5py as h5
        with h5.File("../common_data/fashion_mnist.h5" ,"r") as f:
            train_images = f["train_images"].value
            train_labels = f["train_labels"].value

            test_images = f["test_images"].value
            test_labels = f["test_labels"].value

        def pre_process(data, label):

            data = data/255.0
            data = (data - 0.5) * 2
            if image_size == 32:
                data = np.pad(data, ((0,0),(2,2),(2,2)), "constant", constant_values=(-1,-1))

            data = data[:,::,::].reshape(data.shape[0], -1)
            #data = data.reshape(data.shape[0], -1)
            return data.astype("float32")

        # train_images = train_images[train_labels==8]
        train_images = pre_process(train_images, train_labels)
        test_images = pre_process(test_images, test_labels)
        image_size = np.sqrt(train_images.shape[-1]).astype("int")
        Dx = train_images.shape[-1]


        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_images)),
            batch_size=batch_size, shuffle=True)


    elif dname == "mnist":
        import h5py as h5
        with h5.File("../common_data/mnist.h5" ,"r") as f:
            train_images = f["train_images"].value
            train_labels = f["train_labels"].value

            test_images = f["test_images"].value
            test_labels = f["test_labels"].value

        def pre_process(data, label):

            data = data/255.0
            if binary:
                data = np.floor(data + np.random.rand(*data.shape)).astype("int8")
                if image_size == 32:
                    data = np.pad(data, ((0,0),(2,2),(2,2)), "constant", constant_values=(0,0))
            else:
                data = (data - 0.5) * 2
                if image_size == 32:
                    data = np.pad(data, ((0,0),(2,2),(2,2)), "constant", constant_values=(-1,-1))

            data = data[:,::,::].reshape(data.shape[0], -1)
            #data = data.reshape(data.shape[0], -1)
            return data.astype("float32")

        # train_images = train_images[train_labels==8]
        train_images = pre_process(train_images, train_labels)
        test_images = pre_process(test_images, test_labels)
        image_size = np.sqrt(train_images.shape[-1]).astype("int")
        Dx = train_images.shape[-1]

        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_images)),
            batch_size=batch_size, shuffle=True)

    elif dname == "cifar":

        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        def pre_process(data):
            data = data/255.0
            data = (data-0.5) * 2
            return data.astype("float32")

        ds = []
        for i in range(1,6):
            ds += unpickle("../common_data/cifar-10/data_batch_%d"%i)[b"data"],
        train_images = np.concatenate(ds, 0)
        
        train_images = pre_process(train_images)
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_images)),
            batch_size=batch_size, shuffle=True)

        test_images = unpickle("../common_data/cifar-10/test_batch")[b"data"]
        test_images = pre_process(test_images)
        
        
    elif dname == "celeb":

        def pre_process(data):
            data = data/255.0
            data = (data-0.5) * 2
            data = data.reshape(data.shape[0], -1)
            return data.astype("float32")
        
        images = np.load("../common_data/celeba%d.npz"%image_size)["images"]
        train_images = images[:ntrain]
        
        train_images = pre_process(train_images)
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(train_images)),
            batch_size=batch_size, shuffle=True)

        test_images = images[-ntest:]
        test_images = pre_process(test_images)
        
    
    if return_loader:
        return train_loader, test_images
    else:
        return train_images, test_images


def load_gp(sigma, seed, device = "cuda:0", npoint=100):
    
    from KernelWakeSleep import Kernel
    from Priors import GPPrior

    torch.manual_seed(seed)
    x = torch.linspace(-1,1,npoint,device=device)[:,None]

    cov_kernel = Kernel([torch.nn.Identity()], sigma)
    gp = GPPrior(x, cov_kernel, scale=1.0).to(device)

    rate_fun = lambda x: 0.5 * (1 + torch.erf(x/np.sqrt(2)))

    f_true = gp.sample(1).T
    r_true = rate_fun(f_true)
    y_true = (torch.rand_like(r_true) + r_true).floor()
    return x.numpy().astype("float64"), f_true.numpy().astype("float64"), r_true.numpy().astype("float64"), y_true.numpy().astype("float64")
