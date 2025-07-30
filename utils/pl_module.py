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

 

