import torch
import pl_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import pickle
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix

cert_module = pl_model.ImageCertifier.load_from_checkpoint('./logs/cert_bb_resnet18/resnet18/cifar10/awgn_net/lightning_logs/version_0/checkpoints/saved_epoch=96.ckpt', map_location=torch.device('cpu'))
clf_module = pl_model.ImageClassifier.load_from_checkpoint('./logs/clf/resnet18/cifar10/resnet18_cifar10_k3/lightning_logs/version_0/checkpoints/saved_epoch=45.ckpt', map_location=torch.device('cpu'))

clf = clf_module.clf
cert = cert_module.cert

clf.eval()
cert.eval()

device = torch.device('cuda:3')
clf = clf.to(device)
cert = cert.to(device)

# Load CIFAR-10 test set
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)


clf_preds = []
cert_preds = []
labels = []

def modules_have_same_parameters(model1, model2) -> bool:
    params1 = list(model1.state_dict().items())
    params2 = list(model2.state_dict().items())

    if len(params1) != len(params2):
        return False

    for (name1, param1), (name2, param2) in zip(params1, params2):
        if name1 != name2:
            print(f"Parameter name mismatch: {name1} vs {name2}")
            return False
        if not torch.equal(param1, param2):
            print(f"Parameter values differ at: {name1}")
            return False

    return True

for img, label in tqdm(test_loader):
    img = img.to(device)
    with torch.no_grad():
        out = clf(cert_module.clf_transform(img).to(device)).argmax(dim=-1)
        clf_preds.append(out.item())

    clf1 = clf_module.clf.to(device) 
    clf2 = cert_module.clf.to(device) 

    import pdb 
    pdb.set_trace()

    


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

import pdb 
pdb.set_trace()