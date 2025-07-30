import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import Accuracy
import torchvision
from einops import rearrange
import os

from neuronx_distributed.plugins.lightning import NeuronLTModule
import torch_xla.core.xla_model as xm

class ImageClassifier(NeuronLTModule):
    def __init__(
        self,
        clf: nn.Module,
        num_classes: int,
        optimizer: DictConfig,
        scheduler: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.clf = instantiate(clf)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.clf(x)

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        # Add this line to mark the step for the XLA compiler
        xm.mark_step()

        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()

        loss, preds, y = self.shared_step(batch)

        self.manual_backward(loss)
        opt.step()

        self.train_acc(preds, y)
        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.shared_step(batch)
        self.val_acc(preds, y)
        self.log("val/loss", loss)
        self.log("val/acc", self.val_acc, prog_bar=True)
        return 

    def test_step(self, batch, batch_idx):       
        loss, preds, y = self.shared_step(batch)
        self.test_acc(preds, y)
        self.log('test/loss', loss)
        self.log('test/acc', self.test_acc, prog_bar=True)
        return 

    def configure_optimizers(self):

        optimizer = instantiate(self.hparams.optimizer)
        opt = optimizer(self.parameters())

        if self.hparams.scheduler:
            scheduler = instantiate(self.hparams.scheduler)
            sch = scheduler(opt)
            return {'optimizer': opt, 'lr_scheduler':{'scheduler': sch, 'interval': "epoch"}}

        else:
            return {'optimizer': opt}

 

class ImageCertifier(NeuronLTModule):
    def __init__(
        self,
        cert: nn.Module,
        clf_path: str,
        snr_range: tuple[float, float],
        mc_samples: int,
        optimizer: DictConfig,
        scheduler: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cert = instantiate(cert)

        # Load base classifier to certify
        exp_dir = os.path.abspath(os.path.join(clf_path, "..", ".."))
        conf = OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

        clf_class = get_class(conf.pl_model._target_)
        clf_module = clf_class.load_from_checkpoint(clf_path)

        self.clf = clf_module.clf
        for param in self.clf.parameters():
            param.requires_grad = False

        self.snr_range = snr_range 
        self.mc_samples = mc_samples  
        self.register_buffer("noise", torch.randn(self.mc_samples, 3, 32, 32), persistent=False)

    def forward(self, x, y):
        return self.cert(x, y)

    # img (B, 3, H, W)
    def do_mc(self, img):
        B, C, H, W = img.shape

        img_sqnorms = torch.sum(torch.square(img), dim=(1,2,3))
        snr = torch.rand(B, device=self.device) * (self.snr_range[1] - self.snr_range[0]) + self.snr_range[0] # randomly sample image snr's
        noise_std = torch.sqrt(img_sqnorms / ((C*H*W) * (10**(snr/10)))) # convert image snr to noise standard deviations
                 
        noisy_inputs = rearrange(img, 'B C H W -> B 1 C H W') + rearrange(self.noise, 'M C H W -> 1 M C H W') * rearrange(noise_std, 'B -> B 1 1 1 1')  # (B, M, C, H, W)
        noisy_inputs = rearrange(noisy_inputs, 'B M C H W -> (B M) C H W')

        with torch.no_grad():
            self.clf.eval()
            out = self.clf(noisy_inputs) # (B * mc_samples, num_classes)

        out = rearrange(out, '(B M) N -> B M N', M = self.mc_samples)
        max_idx = torch.argmax(out, dim=-1) 
        one_hot = torch.zeros_like(out, device=self.device)
        one_hot = one_hot.scatter(-1, rearrange(max_idx, 'B M -> B M 1'), 1.0)
        out = torch.mean(one_hot, dim=1)

        # snr = 2*(snr - self.snr_range[0])/(self.snr_range[1] - self.snr_range[0]) - 1 # normalize snr

        return out, noise_std.unsqueeze(-1)


    def training_step(self, batch, batch_idx):
        # 3. Add xm.mark_step()
        xm.mark_step()

        # 2. Implement manual optimization
        opt = self.optimizers()
        opt.zero_grad()

        img, _ = batch
        target, snr = self.do_mc(img)
        out = self.cert(img, snr)
        l1_loss = F.l1_loss(torch.softmax(out, dim=-1), target)

        self.manual_backward(l1_loss)
        opt.step()

        self.log("train/loss", l1_loss, prog_bar=True)
        # Note: You can keep returning the loss, but it's not used for automatic optimization
        return l1_loss

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        target, snr = self.do_mc(img)
        out = self.cert(img, snr)
        l1_loss = F.l1_loss(torch.softmax(out, dim=-1), target)
        self.log("val/loss", l1_loss) 
        return
    
    def test_step(self, batch, batch_idx):
        img, _ = batch
        target, snr = self.do_mc(img)
        out = self.cert(img, snr)
        l1_loss = F.l1_loss(torch.softmax(out, dim=-1), target)
        self.log("test/loss", l1_loss) 
        return

    def configure_optimizers(self):

        optimizer = instantiate(self.hparams.optimizer)
        opt = optimizer(self.parameters())

        if self.hparams.scheduler:
            scheduler = instantiate(self.hparams.scheduler)
            sch = scheduler(opt)
            return {'optimizer': opt, 'lr_scheduler':{'scheduler': sch, 'interval': "epoch"}}

        else:
            return {'optimizer': opt}


# class CertNet(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()

#         # Load ResNet-18 and remove the final classification layer (fc)
#         self.cnn = models.resnet18(pretrained=False)
#         self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)
        
#         self.cnn_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC
#         self.cnn_out_dim = resnet.fc.in_features  # 512 for resnet18

#         # MLP for scalar
#         self.scalar_net = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#         )

#         # Final classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(self.cnn_out_dim + 128, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, image, scalar):
#         """
#         image: Tensor of shape (B, 3, H, W)
#         scalar: Tensor of shape (B, 1)
#         """
#         # ResNet outputs (B, 512, 1, 1), we flatten it to (B, 512)
#         image_features = self.cnn_feature_extractor(image).flatten(1)
#         scalar_features = self.scalar_net(scalar)
#         combined = torch.cat([image_features, scalar_features], dim=1)
#         out = self.classifier(combined)
#         return torch.softmax(out, dim=-1)
