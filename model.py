import torch
from torch import nn

class PaperModel(nn.Module):
    def __init__(self, input_dim=5, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, (5, 5), padding="same")
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 64, (5, 5), padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 8, (3, 70), padding="same")
        self.bn5 = nn.BatchNorm2d(8)
        
        self.conv6 = nn.Conv2d(8, 1, (1, 1), padding="same")
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x of shape (batch, channels, time, freq)
        x = self.conv1(x)  # (batch, 128, time, freq)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)  # (batch, 64, time, freq)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)  # (batch, 64, time, freq)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)  # (batch, 64, time, freq)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)  # (batch, 8, time, freq)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)  # (batch, 1, time, freq)
        
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        
        return x
    
    def predict(self, x):
        x = self.forward(x)
        x = self.sigmoid(x)
        
        return x