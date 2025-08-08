# -*- coding: utf-8 -*-

import torch
from torch import nn


class BasicCoder(nn.Module):
    def __init__(self, data_depth, is_encoder):
        super().__init__()
        self.data_depth = data_depth
        self.is_encoder = is_encoder
        self.block_depth = 32
        self.layers = self._build_models()

    def _build_models(self):

        input_channels = 3 + self.data_depth if self.is_encoder else 3

        layers = nn.ModuleList()
        layers.append(nn.Sequential(
            self._conv2d(input_channels, self.block_depth),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.block_depth),
        ))
        for i in range(4):
            layers.append(nn.Sequential(
                self._conv2d(self.block_depth * (i + 1) + input_channels, self.block_depth,
                    dilation=2 if 0 < i < 3 else 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(self.block_depth),
            ))

        out_channels = 3 if self.is_encoder else self.data_depth

        layers.append(self._conv2d(self.block_depth * (len(layers)) + input_channels, out_channels))

        return layers

    def forward(self, image, data):
        x = torch.cat([image, data], dim=1) if self.is_encoder else image
        x_list = [x]
        for layer in self.layers:
            x = layer(torch.cat(x_list, dim=1))
            x_list.append(x)

        if self.is_encoder:
            x = x + image
        return x

    @staticmethod
    def _conv2d(in_channels, out_channels, kernel_size=3, dilation=1):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation
        )
