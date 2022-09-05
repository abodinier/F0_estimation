import torch
import numpy as np
from tqdm import tqdm

from utils import visualize, evaluate

class Trainer:
    def __init__(self, model, train_data, val_data, loss_cls, optimizer, summary_writer, ckp_path, device="cpu"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_cls = loss_cls
        self.optimizer = optimizer
        self.device = device
        self.summary_writer = summary_writer
        self.ckp_path = ckp_path
        
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
        self.model.to(self.device)
        
        best_loss = np.inf
        
        epoch = 1
        while epoch < n_epochs:
            print(f"EPOCH [{epoch}/{n_epochs}]")
            self.epoch = epoch
            
            train_loss = self.train_step()
            self.summary_writer.add_scalar("train/loss", train_loss, epoch)
            
            val_loss = self.val_step()
            self.summary_writer.add_scalar("val/loss", val_loss, epoch)
            
            if val_loss < best_loss:
                torch.save(self.model.state_dict(), self.ckp_path/"ckp.pt")
            
            print(f"\tTrain loss : {train_loss} | Val loss : {val_loss}")
            
            epoch += 1
        
        print("Training complete.")