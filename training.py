import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualize(model, hcqt, salience):
    predicted_salience = model(hcqt).detach()
    fig = plt.figure(figsize=(10, 10))
    n_examples = 3
    for i in range(n_examples):

        # plot the input
        plt.subplot(n_examples, 3, 1 + (n_examples * i))
        # use channel 1 of the hcqt, which corresponds to h=1
        plt.imshow(hcqt[i, 1, :, :].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f"HCQT")
        plt.axis("tight")

        # plot the target salience
        plt.subplot(n_examples, 3, 2 + (n_examples * i))
        plt.imshow(salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Target")
        plt.axis("tight")

        # plot the predicted salience
        plt.subplot(n_examples, 3, 3 + (n_examples * i))
        plt.imshow(torch.sigmoid(predicted_salience[i, :, :, 0].T), origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Prediction")
        plt.axis("tight")

    plt.tight_layout()
    return fig

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
        self.model.train()
        
        batch_losses = []
        
        for batch in tqdm(self.train_data):
            self.optimizer.zero_grad()
            
            (x, y) = batch
            x.to(self.device)
            y.to(self.device)
            
            out = self.model(x)
            loss = self.loss_cls(out, y)
            batch_losses.append(loss.item())
            
            loss.backward()
            
            self.optimizer.step()
        
        return np.array(batch_losses).mean()
    
    def val_step(self):
        print("here")
        self.model.eval()
        
        batch_losses = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(self.val_data):
                
                (x, y) = batch
                x.to(self.device)
                y.to(self.device)
                
                out = self.model(x)
                loss = self.loss_cls(out, y)
                batch_losses.append(loss.item())
            
        self.summary_writer.add_figure(
            "val/salience_map",
            visualize(self.model, x, y),
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