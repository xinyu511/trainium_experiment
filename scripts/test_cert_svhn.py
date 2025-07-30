import torch
import pl_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import pickle
from torch.utils.data import Subset

cert_module = pl_model.ImageCertifier.load_from_checkpoint('./logs/cert_bb_resnet18/resnet18/svhn/awgn_net/lightning_logs/version_0/checkpoints/saved_epoch=36.ckpt', map_location=torch.device('cpu'))

clf = cert_module.clf
cert = cert_module.cert

clf.eval()
cert.eval()

snr_vals = [-5, -2.5, 0, 2.5, 5, 10]
mc_samples = 100000

device = torch.device('cuda:2')
clf = clf.to(device)
cert = cert.to(device)

# Load CIFAR-10 test set
test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
indices = torch.randperm(len(test_dataset))[:100]
sampled_dataset = Subset(test_dataset, indices.tolist())
test_loader = DataLoader(sampled_dataset, batch_size=1, shuffle=False, num_workers=2)
noise_loader = DataLoader(TensorDataset(torch.randn(mc_samples, 3, 32, 32)), batch_size=1000, shuffle=False, num_workers=2)

for snr in snr_vals:

    errs = []
    for img, _ in tqdm(test_loader):
        B, C, H, W = img.shape
        img_sqnorm = torch.sum(torch.square(img), dim=(1,2,3))
        noise_std = torch.sqrt(img_sqnorm / ((C*H*W) * (10**(snr/10))))

        class_probs = []
        for noise in noise_loader:
            noisy_imgs = img + noise[0] * noise_std
            noisy_imgs = cert_module.clf_transform(noisy_imgs)
            noisy_imgs = noisy_imgs.to(device)

            with torch.no_grad():
                pred = clf(noisy_imgs).argmax(dim=-1)
                class_probs.append(pred)
            
        class_probs = torch.bincount(torch.cat(class_probs), minlength=10)
        class_probs = class_probs.float() / mc_samples
        

        snr_norm = 2*(snr - cert_module.snr_range[0])/(cert_module.snr_range[1] - cert_module.snr_range[0]) - 1
        with torch.no_grad():
            pred_probs = cert(cert_module.clf_transform(img).to(device), torch.ones(1,1, device=device)*snr_norm)
            pred_probs = torch.softmax(pred_probs, dim=-1)

        errs.append(torch.sum(torch.abs(class_probs - pred_probs)).item())

    with open('svhn_results_' + str(snr) + '.pkl', 'wb') as f:
        pickle.dump(errs, f)
    
        
        # add noise

# model = pl_model.ImageCertifier.load_from_checkpoint('./logs/cert/resnet18/cifar10/certifying_resnet18_awgn/lightning_logs/version_1/checkpoints/saved_epoch=25.ckpt')
# mlp = model.cert.mlp
# mlp.cpu()
# mlp.eval()


# out = mlp(torch.ones(1,1)*0.1).detach()
# plt.imshow(out.view(32,32), aspect='auto', interpolation='nearest')
# plt.colorbar()
# plt.savefig('embed.png')

# import pdb 
# pdb.set_trace()