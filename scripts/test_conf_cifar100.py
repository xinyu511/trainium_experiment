import torch
import pl_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import pickle
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix

cert_module = pl_model.ImageCertifier.load_from_checkpoint('./logs/cert_bb_resnet18/resnet18/cifar100/awgn_net/lightning_logs/version_0/checkpoints/saved_epoch=56.ckpt', map_location=torch.device('cpu'))

clf = cert_module.clf
cert = cert_module.cert

clf.eval()
cert.eval()

device = torch.device('cuda:3')
clf = clf.to(device)
cert = cert.to(device)

# Load CIFAR-10 test set
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
indices = torch.randperm(len(test_dataset))[:100]
sampled_dataset = Subset(test_dataset, indices.tolist())
test_loader = DataLoader(sampled_dataset, batch_size=1, shuffle=False, num_workers=2)
noise_loader = DataLoader(TensorDataset(torch.randn(mc_samples, 3, 32, 32)), batch_size=1000, shuffle=False, num_workers=2)


clf_preds = []
cert_preds = []
labels = []

for img, label in tqdm(test_loader):

    with torch.no_grad():
        out = clf(img).argmax(dim=-1)
        clf_preds.append(out.item())
                            

    snr_norm = 1 #2*(snr - cert_module.snr_range[0])/(cert_module.snr_range[1] - cert_module.snr_range[0]) - 1
    with torch.no_grad():
        out = cert(cert_module.clf_transform(img).to(device), torch.ones(1,1, device=device)*snr_norm).argmax(dim=-1)
        cert_preds.append(out.item())

    labels.append(label)


cm_clf = confusion_matrix(labels, clf_preds)
cm_cert = confusion_matrix(labels, cert_preds)

print("CLF Confusion Matrix:")
print(cm_clf)

print("CERT Confusion Matrix:")
print(cm_cert)