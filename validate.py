from models.pl_models import rescale_and_quantize
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.functional.image import peak_signal_noise_ratio
import torch 
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class 
import os
import torch.nn.functional as F
from tqdm import tqdm 
import matplotlib.pyplot as plt 

pl.seed_everything(2024)
ckpt_path = './logs/cmgn_tstepv2_n25/cmgn_20step_n25_tanh_V_b64/lightning_logs/version_0/checkpoints/saved_epoch=9.ckpt'
exp_dir = os.path.abspath(os.path.join(ckpt_path, "..", ".."))
conf = OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

train_ds = instantiate(conf.dataset, data_file=conf.dataparams.train_file, noise_level=conf.dataparams.noise_level/255.0)
val_ds = instantiate(conf.dataset, data_file=conf.dataparams.validation_file, noise_level=conf.dataparams.noise_level/255.0, randomize=False)
test_ds = instantiate(conf.dataset, data_file=conf.dataparams.test_file, noise_level=conf.dataparams.noise_level/255.0, randomize=False)

train_loader = DataLoader(train_ds, batch_size = conf.dataloader.train_batch_size, shuffle = True, pin_memory = True, num_workers = 8)
val_loader = DataLoader(val_ds, batch_size = conf.dataloader.val_batch_size, shuffle = False, pin_memory = True, num_workers = 8)
test_loader = DataLoader(test_ds, batch_size = conf.dataloader.val_batch_size, shuffle = False, pin_memory = True, num_workers = 8)
model_class = get_class(conf.pl_model._target_)
model = model_class.load_from_checkpoint(ckpt_path)

# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:5")
model = model.to(device)
model = model.eval()


iters = 500
step_sizes = torch.linspace(1e-2, 1, steps=5)
lambd_vals = torch.linspace(1e-2, 1, steps=5)
mu_vals = torch.linspace(1e-1, 2, steps=5)
grad_norm_tol = 1e-5

# psnr_vals = torch.zeros(len(test_ds),)
# with torch.no_grad():
#     for i, (y, xstar) in enumerate(test_loader):
                        
#         y = y.to(device)
#         xstar = xstar.to(device)

#         xpred = y - model.forward(y)
#         xpred = rescale_and_quantize(xpred)
#         psnr_vals[i] = peak_signal_noise_ratio(xpred, torch.round(xstar*255), data_range=(0, 255))

#         print('Step: {0}, Avg. PSNR: {1}'.format(i, torch.mean(psnr_vals[0:i+1])))

best_params = {'step':0, 'lambd':0, 'mu':0, 'psnr':0}
with torch.no_grad():
    for step in step_sizes:
        for lambd in lambd_vals:
            for mu in mu_vals:
                psnr_vals = torch.zeros(len(val_ds))
                grad_norms = torch.zeros(len(val_ds))

                for i, (y, xstar) in enumerate(val_loader):
                    
                    y = y.to(device)
                    xstar = xstar.to(device)
                    xpred = y.clone()

                    # do gradient descent
                    xpred = y.detach().clone()
                    for j in range(iters):
                        grad = xpred - y + lambd * model.forward(mu * xpred)
                        grad_norm = torch.linalg.vector_norm(grad.flatten())
                        xpred = xpred - step * grad
                    
                    # track grad norms to update step size
                    # while grad_norm > grad_norm_tol and j <= iters:
                    #     grad = xpred - y + lambd * model.forward(xpred)
                    #     new_grad_norm = torch.linalg.vector_norm(grad.flatten())
                        
                    #     if new_grad_norm > grad_norm:
                    #         step = step*1e-1
                    #     xpred = xpred - step*grad

                    #     grad_norm = new_grad_norm
                    #     j += 1

                    xpred = rescale_and_quantize(xpred)
                    psnr_vals[i] = peak_signal_noise_ratio(xpred, torch.round(xstar*255), data_range=(0, 255))
                    grad_norms[i] = torch.linalg.vector_norm(grad.flatten())

                    print(psnr_vals[i])
                    break
                    # if psnr_vals[i] == torch.amax(psnr_vals):
                    #     best_params['step'] = step
                    #     best_params['lambd'] = lambd 
                    #     best_params['mu'] = mu
                    #     best_params['psnr'] = psnr_vals[i]

                # print('Step: {0}, Lambd: {1}, Mu:{2}, Avg. PSNR: {3}, Max Grad Norm: {4}'.format(step, lambd, mu, torch.mean(psnr_vals), torch.amax(grad_norms)))

print(best_params)
# with torch.no_grad():

#     for step in step_sizes:
#         for lambd in lambd_vals:
#             for mu in mu_vals:

#                 psnr_vals = torch.zeros(len(val_ds))
#                 grad_norms = torch.zeros(len(val_ds))

#                 for i, (y, xstar) in enumerate(val_loader):
                    
#                     y = y.to(device)
#                     xstar = xstar.to(device)


#                     xpred = y.clone()
#                     for j in range(iters):
#                         grad = xpred - y # - lambd * model.forward(mu*xpred) / mu
#                         xpred = xpred - step * grad
#                         xpred = xpred - model.forward(xpred)*F.softplus(model.lambd)

#                         # if torch.linalg.vector_norm(grad.flatten()) < grad_norm_tol:
#                         #     break

#                         # if j%10==0:
#                         #     print(torch.linalg.norm(xpred - xstar))

#                     # print('Done')
#                     xpred = rescale_and_quantize(xpred)
#                     # plt.imshow(xpred.squeeze().cpu(), cmap='grey')
#                     # plt.savefig('test.png')

#                     psnr_vals[i] = peak_signal_noise_ratio(xpred, torch.round(xstar*255), data_range=(0, 255))
#                     grad_norms[i] = torch.linalg.vector_norm(grad.flatten())
#                     break

#                 print('Step: {0}, Lambd: {1}, Mu:{2}, Avg. PSNR: {3}, Max Grad Norm: {4}'.format(step, lambd, mu, torch.mean(psnr_vals[0]), torch.amax(grad_norms)))


# model.requires_grad_(False)
# step = torch.zeros((1), requires_grad=True, device=device)
# lambd = torch.zeros((1), requires_grad=True, device=device)
# mu = torch.zeros((1), requires_grad=True, device=device)
# opt = torch.optim.Adam([lambd], lr = 1e-2)
# loss_fn = model.loss_fn

# for epoch in range(10):

#     avg_loss = 0
#     avg_psnr = 0
#     for i, (y, xstar) in tqdm(enumerate(train_loader)):                      
#         y = y.to(device)
#         xstar = xstar.to(device)
#         xpred = y.clone()

#         xstar_grad = xstar - y + F.softplus(lambd)*model.forward(xstar)
#         loss = model.loss_fn(xstar_grad, torch.zeros_like(xstar_grad))
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#         avg_loss += loss.item()


#     print('Epoch: {0}:Lambd: {1}, Mu:{2}, Avg Loss: {3}'.format(epoch, F.softplus(lambd).item(), F.softplus(mu).item(), avg_loss/len(val_loader)))

# # Test
# iters = 1000
# step_sizes = [1] #torch.linspace(1e-3, 1e-1, steps=10)
# grad_norm_tol = 1e-2
# lambd = F.softplus(lambd)

# with torch.no_grad():
#     psnr_vals = torch.zeros(len(test_ds))
#     grad_norms = torch.zeros(len(val_ds))
    
#     for i, (y, xstar) in enumerate(val_loader):
        
#         y = y.to(device)
#         xstar = xstar.to(device)
#         xpred = y.clone()

#         j = 1
#         step = 1
        
#         # do 1 step of gradient descent
#         xpred = y.detach().clone()
#         grad = xpred - y + lambd * model.forward(xpred)
#         grad_norm = torch.linalg.vector_norm(grad.flatten())
#         xpred = xpred - step * grad
        
#         # track grad norms to update step size
#         while grad_norm > grad_norm_tol and j <= iters:
#             grad = xpred - y + lambd * model.forward(xpred)
#             new_grad_norm = torch.linalg.vector_norm(grad.flatten())
            
#             if new_grad_norm > grad_norm:
#                 step = step*1e-1
#             xpred = xpred - step*grad

#             grad_norm = new_grad_norm
#             j += 1

#         xpred = rescale_and_quantize(xpred)
#         psnr_vals[i] = peak_signal_noise_ratio(xpred, torch.round(xstar*255), data_range=(0, 255))
#         grad_norms[i] = torch.linalg.vector_norm(grad.flatten())

#         print('PSNR: {0}: Grad Norms: {1}'.format(psnr_vals[i], grad_norms[i]))
