import torch
import pl_model
import matplotlib.pyplot as plt 

model = pl_model.ImageCertifier.load_from_checkpoint('./logs/cert/resnet18/cifar10/certifying_resnet18_awgn/lightning_logs/version_1/checkpoints/saved_epoch=25.ckpt')
mlp = model.cert.mlp
mlp.cpu()
mlp.eval()


out = mlp(torch.ones(1,1)*0.1).detach()
plt.imshow(out.view(32,32), aspect='auto', interpolation='nearest')
plt.colorbar()
plt.savefig('embed.png')

import pdb 
pdb.set_trace()