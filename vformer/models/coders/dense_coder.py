# -*- coding: utf-8 -*-

import torch
from torch import nn

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

from vformer.models.coordconv import CoordConv2d

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


class CSPDualBlock(nn.Module):
    def __init__(self, is_encoder, depth, growth_rate, data_depth):
        super(CSPDualBlock, self).__init__()
        self.is_encoder = is_encoder
        self.layers = nn.ModuleList()
        self._build_model(depth, growth_rate, data_depth)

    def _build_model(self, path_depth, growth_rate, data_depth):
        depth = path_depth + data_depth + 3 if self.is_encoder else path_depth + 3
        self.layers.append(
            nn.Sequential(
                _conv2d(depth, growth_rate, kernel_size=3),
                nn.BatchNorm2d(growth_rate),
                _conv2d(growth_rate, growth_rate, kernel_size=3),
                _conv2d(growth_rate, growth_rate, kernel_size=3),
                nn.BatchNorm2d(growth_rate),
                nn.LeakyReLU(),
            )
        )

    def forward(self, image_data, inp):
        x = torch.cat([image_data, inp], dim=1) if inp is not None else image_data
        x = self.layers[0](x)
        return x


class CSPDenseCoder(nn.Module):
    def __init__(self, data_depth, is_encoder):
        super().__init__()
        self.is_encoder = is_encoder
        self.block_count = 4
        self.growth_rate = 48
        self.layers = self._build_models(data_depth=data_depth)

    def _build_models(self, data_depth):

        layers = nn.ModuleList()

        d = 0
        for i in range(self.block_count):
            layers.append(
                CSPDualBlock(self.is_encoder, d, self.growth_rate, data_depth)
            )
            d = (i + 1) * self.growth_rate // 2
            # layers.append(BottleneckBlock(self.is_encoder, block_depth, data_depth))
            # layers.append(ConvNextBlock(self.is_encoder, block_depth, data_depth))

        out_channels = 3 if self.is_encoder else data_depth

        layers.append(
            _conv2d(d * 2 + (3 + data_depth if self.is_encoder else 3), out_channels)
        )

        return layers

    def forward(self, image, data):
        image_data = torch.cat([image, data], dim=1) if self.is_encoder else image
        holdout = [image_data]
        for i in range(self.block_count):
            o = self.layers[i](image_data, x if i > 0 else None)
            h, n = torch.split(o, self.growth_rate // 2, dim=1)
            holdout.append(h)
            if i == 0:
                x = n
            else:
                x = torch.cat([x, n], dim=1)

        x = torch.cat(holdout + [x], dim=1)
        x = self.layers[self.block_count](x)

        if self.is_encoder:
            x = x + image

        return x


def _conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    dilation=1,
    groups=1,
    use_coord=False,
    device=None,
):
    if use_coord:
        return CoordConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            groups=groups,
            device=device,
        )
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2 * dilation,
        dilation=dilation,
        groups=groups,
    )
