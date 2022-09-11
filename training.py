import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import visualize, evaluate
from models import load_from_dir
from data import SalienceDataset
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

class Trainer:
    """Training class in charge of training, checkpoint, loading and performance logging.
    """
    def __init__(self, model, train_data, val_data, loss_cls, optimizer, ckp_path=None, summary_writer=None, device="cpu"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_cls = loss_cls
        self.optimizer = optimizer
        self.device = device
        self.summary_writer = summary_writer
        self.ckp_path = ckp_path
        
        self.init_trainer()
        
    def init_trainer(self):
        self.best_loss = np.inf
        self.epoch = 1
        
    def train_step(self):
        self.model.to(self.device)
        self.model.train()
        
        batch_losses = []
        
        for batch in tqdm(self.train_data):
            self.optimizer.zero_grad()
            
            (x, y) = batch
            x = x.to(self.device)
            y = y.to(self.device)
            
            out = self.model(x)
            loss = self.loss_cls(input=out, target=y)
            batch_losses.append(loss.item())
            
            loss.backward()
            
            self.optimizer.step()
        
        with torch.no_grad():
            train_data_sample = self.train_data.__iter__()._next_data()
            indices = torch.randperm(train_data_sample[0].size(0))[:32]
            xtrain, ytrain = train_data_sample[0][indices], train_data_sample[1][indices]
            
            metrics = evaluate(self.model, (xtrain, ytrain))
            for metric_name, values in metrics.items():
                self.summary_writer.add_scalar(
                    f"train/{metric_name}",
                    np.mean(values),
                    self.epoch
                )
        
        return np.array(batch_losses).mean()
    
    def val_step(self):
        self.model.to(self.device)
        self.model.eval()
        
        batch_losses = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(self.val_data):
                
                (x, y) = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                out = self.model(x)
                loss = self.loss_cls(input=out, target=y)
                batch_losses.append(loss.item())
        
            val_data_sample = self.val_data.__iter__()._next_data()
            indices = torch.randperm(val_data_sample[0].size(0))[:32]
            xval, yval = val_data_sample[0][indices], val_data_sample[1][indices]
            
            metrics = evaluate(self.model, (xval, yval))
            for metric_name, values in metrics.items():
                self.summary_writer.add_scalar(
                    f"val/{metric_name}",
                    np.mean(values),
                    self.epoch
                )
            
            self.summary_writer.add_figure(
                "val/salience_map",
                visualize(self.model, xval, yval),
                self.epoch
            )
        
        return np.array(batch_losses).mean()

    def train(self, n_epochs):
        """Train method

        Args:
            n_epochs (int): Number of epochs
        """
        self.model.to(self.device)
        
        while self.epoch < n_epochs:
            print(f"EPOCH [{self.epoch}/{n_epochs}]")
            
            train_loss = self.train_step()
            self.summary_writer.add_scalar("train/loss", train_loss, self.epoch)
            
            val_loss = self.val_step()
            self.summary_writer.add_scalar("val/loss", val_loss, self.epoch)
            
            self.checkpoint(val_loss)
            
            print(f"\tTrain loss : {train_loss} | Val loss : {val_loss}")
            
            self.epoch += 1
        
        print("Training complete.")
    
    def checkpoint(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            ckp = {
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
                "model": self.model.state_dict(),
                "best_loss": self.best_loss
            }
            torch.save(ckp, self.ckp_path)
        else:
            pass
    
    def load_from_checkpoint(self, ckp):      
        self.optimizer.load_state_dict(ckp["optimizer"])
        self.model.load_state_dict(ckp["model"])
        self.epoch = ckp["epoch"]
        self.best_loss = ckp["best_loss"]
        
        return self
    
    @staticmethod
    def load_from_cfg(cfg):
        """Create a Trainer instance from config file

        Args:
            cfg (dict): Config file

        Returns:
            Trainer: Trainer instance
        """
        model_path = cfg["MODEL_PATH"]
        model = load_from_dir(model_path)
        
        train_data = SalienceDataset(cfg["TRAIN_DATA_DIR"], ratio=cfg["DATA_RATIO"])
        val_data = SalienceDataset(cfg["VALIDATION_DATA_DIR"], ratio=cfg["DATA_RATIO"])
        train_loader = DataLoader(train_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=cfg["BATCH_SIZE"], shuffle=True)
        
        loss_cls = BCEWithLogitsLoss()
        optimizer = Adam(lr=cfg["LR"], params=model.parameters(), weight_decay=cfg["WEIGHT_DECAY"])
        
        device = cfg["DEVICE"]
        
        summary_writer = SummaryWriter(cfg["TENSORBOARD_DIR"])
        ckp_path = cfg["CKP_PATH"]
        ckp = torch.load(ckp_path)
        
        instance = Trainer(
            model=model,
            train_data=train_loader,
            val_data=val_loader,
            loss_cls=loss_cls,
            optimizer=optimizer,
            device=device,
            summary_writer=summary_writer,
            ckp_path=ckp_path
        ).load_from_checkpoint(ckp)
        
        return instance