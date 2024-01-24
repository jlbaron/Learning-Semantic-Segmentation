'''
downsample -> upsample
while sending copy and crops to upsample
downsample with conv, relu, and max pool
upsample with transpose conv

Used code from: https://nn.labml.ai/unet/index.html
I also formatted the images to be the same size so nothing changes
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 2 convolution layers with relu
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device='cpu'):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, device=device)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=device)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

# max pool to reduce size
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)

# transpose conv to increase size
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device='cpu'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, device=device)

    def forward(self, x: torch.Tensor):
        return self.up(x)

# add sample from down path to up path to maintain spacial details about images
class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x

# UNet downsamples from 512,512 to 35,35 then back up to original 512, 512
# output as many filters as classes to label
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device='cpu'):
        super().__init__()

        self.down_conv = nn.ModuleList([DoubleConvolution(i, o, device) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(512, 1024, device)
        self.up_sample = nn.ModuleList([UpSample(i, o, device) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.up_conv = nn.ModuleList([DoubleConvolution(i, o, device) for i, o in
                                     [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, device=device)

    def forward(self, x: torch.Tensor):
        pass_through = []

        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)
        return x
