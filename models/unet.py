'''
downsample -> upsample
while sending copy and crops to upsample
downsample with conv, relu, and max pool
upsample with transpose conv
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: untested from chatgpt, will modify once dataset is ready
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Contracting Path
        for _ in range(5):
            self.encoder.append(self.conv_block(in_channels, n_filters, batchnorm))
            in_channels = n_filters
            n_filters *= 2
            self.encoder.append(nn.Dropout2d(dropout))
            self.encoder.append(self.pool)

        # Bottleneck
        self.bottleneck = self.conv_block(in_channels, n_filters, batchnorm)

        # Expansive Path
        for _ in range(4):
            self.decoder.append(self.upsample)
            self.decoder.append(self.conv_block(in_channels + n_filters, n_filters // 2, batchnorm))
            in_channels //= 2
            n_filters //= 2
            self.decoder.append(nn.Dropout2d(dropout))

        self.decoder.append(nn.Conv2d(n_filters, out_channels, kernel_size=1))

    def conv_block(self, in_channels, out_channels, batchnorm):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []

        # Encoding
        for module in self.encoder:
            x = module(x)
            if isinstance(module, nn.MaxPool2d):
                skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding
        for module in self.decoder:
            if isinstance(module, nn.Upsample):
                x = module(x)
                skip_connection = skip_connections.pop()
                x = torch.cat([x, skip_connection], dim=1)
            else:
                x = module(x)

        return x

# Usage:
# Create the model with input and output channels
model = UNet(in_channels=3, out_channels=1)  # Adjust input and output channels accordingly
