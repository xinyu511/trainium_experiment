import torch
import pl_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import pickle
from torch.utils.data import Subset
import time 
import numpy as np
import pytorch_lightning as pl
pl.seed_everything(12345)

cert_module = pl_model.ImageCertifier.load_from_checkpoint('./logs/cert_bb_resnet18/resnet18/cifar10/awgn_net/lightning_logs/version_0/checkpoints/saved_epoch=65.ckpt', map_location=torch.device('cpu'))

clf = cert_module.clf
cert = cert_module.cert

clf.eval()
cert.eval()

snr = 0
snr_norm = 2*(snr - cert_module.snr_range[0])/(cert_module.snr_range[1] - cert_module.snr_range[0]) - 1
mc_samples = 100
clf = clf.cpu()
cert = cert.cpu()

# Load CIFAR-10 test set
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
indices = torch.randperm(len(test_dataset))[:1000]
sampled_dataset = Subset(test_dataset, indices.tolist())
test_loader = DataLoader(sampled_dataset, batch_size=1, shuffle=False, num_workers=2)
noise_loader = DataLoader(TensorDataset(torch.randn(mc_samples, 3, 32, 32)), batch_size=100000, shuffle=False, num_workers=2)


mc_times = []
nn_times = []
for img, _ in tqdm(test_loader):
    B, C, H, W = img.shape
    start_cp = time.perf_counter()
    img_sqnorm = torch.sum(torch.square(img), dim=(1,2,3))
    noise_std = torch.sqrt(img_sqnorm / ((C*H*W) * (10**(snr/10))))
    class_probs = []
    for noise in noise_loader:
        noisy_imgs = img + noise[0] * noise_std
        noisy_imgs = cert_module.clf_transform(noisy_imgs)
  
        with torch.no_grad():
            pred = clf(noisy_imgs).argmax(dim=-1)
            class_probs.append(pred)
        
    class_probs = torch.bincount(torch.cat(class_probs), minlength=10)
    class_probs = class_probs.float() / mc_samples
    end_cp = time.perf_counter()
    mc_times.append(end_cp - start_cp)


    
    start_cp = time.perf_counter()
    with torch.no_grad():
        pred_probs = cert(cert_module.clf_transform(img), torch.ones(1,1)*snr_norm)
        pred_probs = torch.softmax(pred_probs, dim=-1)
    end_cp = time.perf_counter()
    nn_times.append(end_cp - start_cp)

print('MC TIMES')
print(np.mean(mc_times))
print(np.std(mc_times))

print()
print('NN TIMES')
print(np.mean(nn_times))
print(np.std(nn_times))


   