import torch
import torchvision
from torchvision.models import swin_transformer
from torch import nn, Tensor
from torch.nn.functional import leaky_relu
from functools import partial
from typing import List
import torch

from torchvision.models.swin_transformer import SwinTransformerBlockV2 as swinv2


class CoderHeadBlock(nn.Module):
    def __init__(self, args, inp_channels, channels, out_channels, resolution):
        super().__init__()
        self._build_models(args, inp_channels, channels, out_channels, resolution)

    def _build_models(self, args, inp_channels, channels, out_channels, resolution):
        self.transformer = SwinTransBlock(args, input_channels=inp_channels, channels=channels,
                                          block_depth=1, resolution=resolution)

        self.seq = nn.Sequential(
            # nn.Conv2d(channels, channels // 2, kernel_size=1),
            # nn.Conv2d(channels // 2, channels // 2, kernel_size=7, padding=3, groups=channels // 2),
            # # nn.LayerNorm(inp_channels//2),
            # nn.BatchNorm2d(channels // 2),
            # nn.Conv2d(channels // 2, channels // 4, kernel_size=1),
            # nn.Conv2d(channels // 4, channels // 4, kernel_size=7, padding=3, groups=channels // 4),
            # nn.LeakyReLU(),
            # # nn.LayerNorm(inp_channels//4),
            # nn.BatchNorm2d(channels // 4),
            # nn.Conv2d(channels // 4, channels // 8, kernel_size=1),
            # nn.Conv2d(channels // 8, channels // 8, kernel_size=7, padding=3, groups=channels // 8),
            # nn.Conv2d(channels // 8, out_channels, kernel_size=1),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.seq(self.transformer(x, None))


# class SwinTransBlock(nn.Module):
#     def __init__(self, input_channels, channels, block_depth, resolution=360):
#         super(SwinTransBlock, self).__init__()
#         self.block_depth = block_depth
#         self.layers = nn.ModuleList()
#         self._build_model(input_channels, channels, block_depth, resolution)
#
#     def _build_model(self, input_channels, channels, block_depth, resolution):
#         # depth = path_depth + data_depth + 3 if self.is_encoder else path_depth + 3
#         for i in range(block_depth):
#             self.layers.append(nn.Conv2d(in_channels=input_channels, out_channels=channels, kernel_size=1))
#             self.layers.append(nn.BatchNorm2d(channels))
#             self.layers.append(Permute([0, 2, 3, 1]))
#             self.layers.append(
#                 swinv2(dim=channels, num_heads=4, window_size=[8, 8], shift_size=[0, 0]))  # Non-shifted window
#             self.layers.append(
#                 swinv2(dim=channels, num_heads=4, window_size=[8, 8], shift_size=[4, 4]))  # Shifted window
#             self.layers.append(Permute([0, 3, 1, 2]))
#             self.layers.append(nn.BatchNorm2d(channels))
#
#     def forward(self, image_data, inp):
#         x = torch.cat([image_data, inp], dim=1) if inp is not None else image_data
#         for layer in self.layers:
#             x = layer(x)
#         return x


class TransCSPDenseCoder(nn.Module):
    def __init__(self, data_depth, patch_size, is_encoder, resolution):
        super().__init__()
        self.is_encoder = is_encoder
        self.patch_size = patch_size
        self.patch_multi = patch_size * patch_size
        self.block_count = 4
        self.growth_rate = 48 * self.patch_multi
        self.layers = self._build_models(data_depth=data_depth, resolution=resolution)

    def _build_models(self, data_depth, resolution):
        layers = nn.ModuleList()
        initial_chann = data_depth + 3 if self.is_encoder else 3
        self.first_conv = nn.Conv2d(initial_chann * self.patch_multi, 24 * self.patch_multi, kernel_size=3, padding=1)
        d = (initial_chann + 24) * self.patch_multi
        for i in range(self.block_count):
            layers.append(
                SwinTransBlock(input_channels=d, channels=self.growth_rate, block_depth=1, resolution=resolution)
            )
            d += self.growth_rate // 2

        # out_channels = 3 if self.is_encoder else data_depth
        # layers.append(_conv2d(d * 2 + (3 + data_depth if self.is_encoder else 3), out_channels))
        d += self.block_count * (self.growth_rate // 2)
        layers.append(
            CoderHeadBlock(d, self.growth_rate * self.block_count,
                           (3 if self.is_encoder else data_depth) * self.patch_multi, resolution)
        )

        return layers

    def forward(self, image, data):
        image_data = torch.cat([image, data], dim=1) if self.is_encoder else image
        # orig_em = self.im_embedding(image_data)
        # conv_em = self.conv_embedding(self.first_conv(image_data))
        # first_cat = torch.cat([orig_em, conv_em], dim=1)
        image_data = create_windows(image_data, self.patch_size)
        first_c = self.first_conv(image_data)
        first_cat = torch.cat([image_data, first_c], dim=1)
        holdout = [first_cat]
        x = None
        for i in range(self.block_count):
            o = self.layers[i](first_cat, x if i > 0 else None)
            h, n = torch.split(o, self.growth_rate // 2, dim=1)
            holdout.append(h)
            if i == 0:
                x = n
            else:
                x = torch.cat([x, n], dim=1)

        x = torch.cat(holdout + [x], dim=1)
        x = self.layers[self.block_count](x)

        x = revert_windows(x, self.patch_size)

        # if self.is_encoder:
        #     x = x + image

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


def _patch_embedding(patch_size, in_channels):
    # split image into non-overlapping patches (Partitioning and Patch Embedding)
    return nn.Conv2d(in_channels, in_channels * patch_size * patch_size, kernel_size=(patch_size, patch_size),
                     stride=(patch_size, patch_size))


class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        return self.transposed_conv(x)

    # ===================================== WINDOWED CREATION ==========================


def create_windows(x: Tensor, patch_size: int):
    # Example input tensor [n, c, h, w] where h and w are divisible by 4
    n, c, h, w = x.size()
    x = x.view(n, c, h // patch_size, patch_size, w // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(n, c * patch_size * patch_size, h // patch_size, w // patch_size)
    return x


def revert_windows(x: Tensor, patch_size: int):
    n, c, h, w = x.size()
    c = c // patch_size // patch_size
    w *= patch_size
    h *= patch_size
    x = x.view(n, c, patch_size, patch_size, h // patch_size, w // patch_size)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.contiguous().view(n, c, h, w)
    return x


class SwinTransBlock(nn.Module):
    def __init__(self, args, input_channels, block_depth, patch_size=4, resolution=360, dropout=0.):
        super(SwinTransBlock, self).__init__()
        self.block_depth = block_depth
        self.patch_size = patch_size
        self.layers = nn.ModuleList()
        self._build_model(input_channels * patch_size * patch_size, block_depth, dropout)

    def _build_model(self, input_channels, block_depth, dropout):
        # depth = path_depth + data_depth + 3 if self.is_encoder else path_depth + 3
        for i in range(block_depth):
            # self.layers.append(nn.Conv2d(in_channels=input_channels, out_channels=channels, kernel_size=1))
            # self.layers.append(nn.BatchNorm2d(channels))
            self.layers.append(Permute([0, 2, 3, 1]))
            self.layers.append(
                swinv2(dim=input_channels, num_heads=4, window_size=[8, 8], shift_size=[0, 0]))  # Non-shifted window
            self.layers.append(
                swinv2(dim=input_channels, num_heads=4, window_size=[8, 8], shift_size=[4, 4]))  # Shifted window
            self.layers.append(Permute([0, 3, 1, 2]))
            self.layers.append(nn.BatchNorm2d(input_channels))
            if dropout:
                self.layers.append(nn.Dropout2d(dropout))

    def forward(self, x):
        x = create_windows(x, self.patch_size)
        inp = x
        for layer in self.layers:
            x = layer(x)
        x = x + inp
        x = revert_windows(x, self.patch_size)
        return x
