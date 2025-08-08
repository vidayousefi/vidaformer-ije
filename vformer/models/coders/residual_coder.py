# -*- coding: utf-8 -*-

import torch
from torch import nn


# class BottleneckBlock(nn.Module):
#     def __init__(self, is_encoder, depth, data_depth):
#         super(BottleneckBlock, self).__init__()
#         self.is_encoder = is_encoder
#         self.layers = nn.ModuleList()
#         self._build_model(depth, data_depth)
#
#     def _build_model(self, path_depth, data_depth):
#         depth = path_depth + data_depth + 3 if self.is_encoder else path_depth + 3
#         self.layers.append(nn.Sequential(
#             # _conv2d(depth, depth, kernel_size=7, groups=depth),
#             _conv2d(depth, depth, kernel_size=3),
#             nn.BatchNorm2d(depth),
#             # nn.LeakyReLU(),
#             _conv2d(depth, path_depth // 2, kernel_size=3),
#             nn.BatchNorm2d(path_depth//2),
#             nn.LeakyReLU(),
#             _conv2d(path_depth//2, path_depth, kernel_size=3),
#             # nn.LeakyReLU(),
#         ))
#
#     def forward(self, image, inp, data):
#         x = torch.cat([image, inp, data], dim=1) if self.is_encoder else torch.cat([image, inp], dim=1)
#         x = self.layers[0](x)
#         x = x + inp
#         return x


# class ConvNextBlock(nn.Module):
#     def __init__(self, is_encoder, depth, data_depth):
#         super(ConvNextBlock, self).__init__()
#         self.is_encoder = is_encoder
#         self.layers = nn.ModuleList()
#         self._build_model(depth, data_depth)
#
#     def _build_model(self, path_depth, data_depth):
#         depth = path_depth + data_depth if self.is_encoder else path_depth
#         self.layers.append(nn.Sequential(
#             _conv2d(depth, depth, kernel_size=7, groups=depth),
#             nn.BatchNorm2d(depth),
#             _conv2d(depth, path_depth * 2, kernel_size=1),
#             nn.LeakyReLU(),
#             _conv2d(path_depth * 2, path_depth, kernel_size=1),
#         ))
#
#     def forward(self, image, inp, data):
#         x = torch.cat([inp, data], dim=1) if self.is_encoder else inp
#         x = self.layers[0](x)
#         x = x + inp
#         return x
from torch.nn.functional import leaky_relu


class DualBlock(nn.Module):
    def __init__(self, is_encoder, depth, data_depth):
        super(DualBlock, self).__init__()
        self.is_encoder = is_encoder
        self.layers = nn.ModuleList()
        self._build_model(depth, data_depth)

    def _build_model(self, path_depth, data_depth):
        depth = path_depth + data_depth + 3 if self.is_encoder else path_depth + 3
        indepth=depth//2
        self.layers.append(nn.Sequential(
            # _conv2d(depth, depth, kernel_size=7, groups=depth),
            _conv2d(depth, indepth, kernel_size=1),
            nn.BatchNorm2d(indepth),
            # nn.LeakyReLU(),
            _conv2d(indepth, indepth, kernel_size=3),
            _conv2d(indepth, indepth, kernel_size=3),
            nn.BatchNorm2d(indepth),
            nn.LeakyReLU(),
            _conv2d(indepth, path_depth, kernel_size=1),
            # nn.LeakyReLU(),

        ))

    def forward(self, image, inp, data):
        x = torch.cat([image, inp, data], dim=1) if self.is_encoder else torch.cat([image, inp], dim=1)
        x = self.layers[0](x)
        x = x + inp
        x = leaky_relu(x)
        return x


class ResidualCoder(nn.Module):
    def __init__(self, data_depth, is_encoder):
        super().__init__()
        self.is_encoder = is_encoder
        self.block_count = 3
        self.layers = self._build_models(block_depth=64, data_depth=data_depth)

    def _build_models(self, block_depth, data_depth):

        layers = nn.ModuleList()

        layers.append(_conv2d(3, block_depth, kernel_size=1))

        for i in range(self.block_count):
            layers.append(DualBlock(self.is_encoder, block_depth, data_depth))
            # layers.append(BottleneckBlock(self.is_encoder, block_depth, data_depth))
            # layers.append(ConvNextBlock(self.is_encoder, block_depth, data_depth))

        out_channels = 3 if self.is_encoder else data_depth

        layers.append(_conv2d(block_depth, out_channels))

        return layers

    def forward(self, image, data):
        x = image
        x = self.layers[0](x)
        for i in range(self.block_count):
            x = self.layers[i + 1](image, x, data)
        x = self.layers[self.block_count + 1](x)

        if self.is_encoder:
            x = x + image
        return x


def _conv2d(in_channels, out_channels, kernel_size=3, dilation=1, groups=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2 * dilation,
        dilation=dilation,
        groups=groups
    )
