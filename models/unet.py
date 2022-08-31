import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_KERNEL_SIZE = (3, 3)

class DownConv(nn.Module):
    def __init__(self, input_channels, output_channels, k=DEFAULT_KERNEL_SIZE):
        super().__init__()
        self.conv = Conv(input_channels, output_channels, k=k)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        return x
    
    
class UpConv(nn.Module):
    def __init__(self, input_channels, output_channels, k=DEFAULT_KERNEL_SIZE, bilinear_interp=True):
        super().__init__()
        self.bilinear_interp = bilinear_interp

        self.up = nn.ConvTranspose2d(input_channels // 2, input_channels // 2, 2, stride=2)

        self.conv = Conv(input_channels, output_channels, k=k)

    def forward(self, x_small, x_big):
        if self.bilinear_interp:
            x_small = F.interpolate(x_small, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x_small = self.up(x_small)

        # Align shapes
        diffY = x_big.size()[2] - x_small.size()[2]
        diffX = x_big.size()[3] - x_small.size()[3]

        x_small = F.pad(
            x_small,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2
            ]
        )

        x = torch.cat([x_big, x_small], dim=1)
        x = self.conv(x)
        return x
    
    
class Conv(nn.Module):
    def __init__(self, input_channels, output_channels, k=(3, 3)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, k, padding="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channels):
        super(UNet, self).__init__()
        self.head = Conv(input_channels, 32, k=(10, 5))

        self.down1 = DownConv(32, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 256)

        self.up4 = UpConv(512, 128)
        self.up3 = UpConv(256, 64)
        self.up2 = UpConv(128, 32)
        self.up1 = UpConv(64, 32)

        self.tail = Conv(32, 1)

    def forward(self, x):
        x0 = self.head(x)

        d1 = self.down1(x0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up4(d4, d3)
        u3 = self.up3(u4, d2)
        u2 = self.up2(u3, d1)
        u1 = self.up1(u2, x0)

        x = self.tail(u1)
        
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)

        return x


if __name__ == "__main__":
    import numpy as np
    x = torch.tensor(np.load("/Users/alexandre/mir_datasets/medleydb_pitch/DATA/HCQT_SALIENCE/small_train/0-136.npz")["hcqt"][np.newaxis, :, :, :])
    model = UNet(x.shape[1])
    out = model(x)
    out[0, 0, :, :]
    print(out.shape)