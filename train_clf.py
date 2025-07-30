from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate 
import os 

pl.seed_everything(12345)

def save_hydra_config(cfg, path):
    os.makedirs(path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(path, "config.yaml"))

@hydra.main(config_path="configs/clf/", config_name="config.yaml")
def main(conf: DictConfig):
    
    train_ds = instantiate(conf.dataset.train_dataset)
    train_ds, val_ds = random_split(train_ds, [conf.dataset.train_split, conf.dataset.val_split])
    test_ds = instantiate(conf.dataset.test_dataset)

    train_loader = DataLoader(train_ds, batch_size = conf.dataloader.train_batch_size, shuffle = True, pin_memory = True, num_workers = 8)
    val_loader = DataLoader(val_ds, batch_size = conf.dataloader.val_batch_size, shuffle = False, pin_memory = True, num_workers = 8)
    test_loader = DataLoader(test_ds, batch_size = conf.dataloader.test_batch_size, shuffle = False, pin_memory = True, num_workers = 8)
    
    model = instantiate(conf.pl_model, _recursive_=False)    

    tb_logger = pl.loggers.TensorBoardLogger(conf.trainer.log_dir)
    trainer = pl.Trainer(
            accelerator = 'gpu',
            devices = [conf.trainer.gpu],
            accumulate_grad_batches=conf.trainer.accumulate_grad_batches,
            gradient_clip_val=conf.trainer.gradient_clip_val,
            max_epochs = conf.trainer.max_epochs,
            log_every_n_steps = conf.trainer.log_every_n_steps,
            logger = tb_logger,
            val_check_interval = conf.trainer.val_check_interval,
            deterministic = conf.trainer.deterministic,
            num_sanity_val_steps = 1, # Change when testing code modifications
            callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'),
                        pl.callbacks.ModelCheckpoint(filename = 'saved_{epoch}',
                                                                    save_top_k = 1,
                                                                    monitor = conf.trainer.val_monitor,
                                                                    mode = conf.trainer.mode,
                                                                    save_last = False
                                                                    )])
    trainer.fit(model, train_loader, val_loader)
    save_hydra_config(conf, trainer.logger.log_dir)
    trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":
    main()