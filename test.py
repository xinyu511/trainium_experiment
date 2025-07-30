from models.pl_models import rescale_and_quantize
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.functional.image import peak_signal_noise_ratio
import torch 
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class 
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm 

pl.seed_everything(2024)

ckpt_path = './logs/cmgn_fp/cmgn_n25/lightning_logs/version_0/checkpoints/saved_epoch=49.ckpt'
exp_dir = os.path.abspath(os.path.join(ckpt_path, "..", ".."))
conf = OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

ckpt_path = './logs/cmgn_fp/cmgn_n25/lightning_logs/version_0/checkpoints/saved_epoch=49.ckpt'
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
    device = torch.device("cuda:4")
model = model.to(device)
model = model.eval()



max_iters = 100

with torch.no_grad():

#     print('Finding best step size')
#     step_vals = np.logspace(-2, 1, 50)
    
#     max_psnr = 0
#     best_step = 0
#     for s in tqdm(step_vals):
#         psnr_step = 0
#         for i, (y, xstar) in enumerate(val_loader):

#             y = y.to(device)
#             xstar = xstar.to(device)
#             xpred = y.clone()

#             for j in range(max_iters):
#                 grad = model.forward(xpred, y)
#                 xpred = xpred - s * grad

#             xhat = rescale_and_quantize(xpred)
#             psnr_step += peak_signal_noise_ratio(xhat, xstar*255, data_range=(0, 255))

#         psnr_step = psnr_step / len(val_ds)
#         if psnr_step > max_psnr:
#             max_psnr = psnr_step
#             best_step = s


#     print(max_psnr)
#     print('Found Best Step Size: ', best_step)


    # print('Testing...\n')
    # test_psnr = 0
    # for i, (y, xstar) in tqdm(enumerate(test_loader), total=len(test_loader)):
    #     y = y.numpy()
    #     xhat = denoise_tv_chambolle(y, weight=best_w, max_num_iter=500)
    #     xhat = torch.from_numpy(xhat)
    #     xhat = rescale_and_quantize(xhat)
    #     test_psnr += peak_signal_noise_ratio(xhat, xstar*255, data_range=(0, 255))

    # print('\n\n')
    # print('Avg Test PSNR: ')
    # print(test_psnr / len(test_ds))




    # tol = 1e-6
    # step_size = 1e-1
    psnr_vals = []
    max_iters = 100
    step_size = 3.24

    for i, (y, xstar) in enumerate(test_loader):
        
        y = y.to(device)
        xstar = xstar.to(device)

        y = model.transform(y)

        # do 1 step of gradient descent
        xpred = y.clone()
        grad = model.forward(xpred, y)
        grad_norm = torch.linalg.vector_norm(grad.flatten())
        xpred = xpred - step_size * grad
        
        # track grad norms to update step size
        j = 0
        while j < max_iters:
            grad = model.forward(xpred, y)
            new_grad_norm = torch.linalg.vector_norm(grad.flatten())
            
            # if new_grad_norm > grad_norm:
            #     step_size = step_size*0.1

            xpred = xpred - step_size*grad
            grad_norm = new_grad_norm
           
            xhat = rescale_and_quantize(xpred)
            psnr_val = peak_signal_noise_ratio(xhat, xstar*255, data_range=(0, 255))
            # print(psnr_val)
            j += 1
                

            # print(new_grad_norm)
        
        # input()
        # plt.figure()
        # plt.imshow(xpred.cpu()[0,0], cmap='gray')
        # plt.savefig('pred2.png')
        # plt.close()
        # print('Done')
        
        # import pdb 
        # pdb.set_trace()
        # import pdb 
        # pdb.set_trace()   
        # input()
        psnr_vals.append(psnr_val.item())
        print('PSNR: ', psnr_val.item(), 'Iters: ', j, 'Grad Norm: ', grad_norm.item(), 'Avg: ', sum(psnr_vals)/len(psnr_vals))

    
    


print(np.mean(psnr_vals))