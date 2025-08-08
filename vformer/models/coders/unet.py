import torch
import torch.nn.functional as F
from torch import nn

from vformer.models.coders.dense_coder import _conv2d
from vformer.models.coders.sade import SwinTransBlock


class UnetConvBlock(nn.Module):
    def __init__(
        self, opt, in_channels, channels, device, inverse_bottleneck, dropout: float
    ):
        super().__init__()
        self.projection = None
        if in_channels != channels:
            self.projection = _conv2d(in_channels, channels, kernel_size=1)

        if inverse_bottleneck:
            inv_multi = 4
            self.layers = nn.Sequential(
                _conv2d(
                    channels,
                    channels * inv_multi,
                    kernel_size=1,
                    device=device,
                    use_coord=opt.coordconv,
                ),
                nn.BatchNorm2d(channels * inv_multi),
                _conv2d(
                    channels * inv_multi,
                    channels * inv_multi,
                    kernel_size=3,
                    device=device,
                    use_coord=opt.coordconv,
                ),
                _conv2d(
                    channels * inv_multi,
                    channels,
                    kernel_size=1,
                    device=device,
                    use_coord=opt.coordconv,
                ),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(),
            )
        else:
            self.layers = nn.Sequential(
                _conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    device=device,
                    use_coord=opt.coordconv,
                ),
                nn.BatchNorm2d(channels),
                _conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    device=device,
                    use_coord=opt.coordconv,
                ),
                _conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    device=device,
                    use_coord=opt.coordconv,
                ),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(),
            )
        if dropout:
            self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        if self.projection:
            x = self.projection(x)
        return x + self.layers(x)


class HybridUnet(nn.Module):
    def __init__(
        self,
        opt,
        is_encoder,
        data_depth,
        encoder_blocks,
        decoder_blocks,
        base_channels,
        device,
        inverse_bottleneck,
        dropout,
    ):
        super().__init__()
        self.is_encoder = is_encoder
        self.blocks = encoder_blocks
        self.data_depth = data_depth
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks[::-1]
        self.base_channels = base_channels
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self._build_model(opt, is_encoder, device, inverse_bottleneck, dropout)

    def _build_model(self, opt, is_encoder, device, inverse_bottleneck, dropout):
        inp_ch = self.data_depth + 3 if is_encoder else 3
        shortcuts = []
        stage_idx = 0
        curr_ch = 0
        for s in self.encoder_blocks:
            stage_blocks = []
            curr_ch = (
                self.base_channels * (2**stage_idx)
                if stage_idx < len(self.encoder_blocks) - 1
                else curr_ch
            )
            for b in range(s):
                if stage_idx < len(self.encoder_blocks) - 1 or not opt.transformer:
                    stage_blocks.append(
                        UnetConvBlock(
                            opt, inp_ch, curr_ch, device, inverse_bottleneck, dropout
                        )
                    )
                    inp_ch = curr_ch
                else:
                    stage_blocks.append(
                        SwinTransBlock(
                            opt, inp_ch, block_depth=1, patch_size=2, dropout=dropout
                        )
                    )
            shortcuts.append(curr_ch)
            self.encoder_layers.append(nn.Sequential(*stage_blocks))
            stage_idx += 1
        stage_idx -= 2

        for s in self.decoder_blocks:
            stage_blocks = []
            curr_ch = self.base_channels * (2**stage_idx)
            stage_blocks.append(
                _conv2d(shortcuts[stage_idx] + shortcuts[stage_idx + 1], curr_ch, 1)
            )
            for b in range(s):
                stage_blocks.append(
                    UnetConvBlock(
                        opt,
                        curr_ch,
                        curr_ch,
                        device,
                        inverse_bottleneck,
                        dropout if b < s - 1 else 0,
                    )
                )
            self.decoder_layers.append(nn.Sequential(*stage_blocks))
            stage_idx -= 1

        self.last_conv = _conv2d(curr_ch, 3 if self.is_encoder else self.data_depth, 1)

    def forward(self, x):
        image = x[:, :3]
        stage_outputs = []
        stage_idx = 0
        for stage in self.encoder_layers:
            x = stage(x)
            stage_outputs.append(x)
            if stage_idx < len(self.encoder_layers) - 1:
                x = F.max_pool2d(x, kernel_size=2)
                stage_idx += 1
        stage_idx -= 1
        for stage in self.decoder_layers:
            previous = stage_outputs[stage_idx]
            x = F.interpolate(
                x, size=(previous.size(2), previous.size(3)), mode="nearest"
            )
            x = stage(torch.cat([previous, x], dim=1))
            stage_idx -= 1

        x = self.last_conv(x)

        if self.is_encoder:
            x = x + image
        return x
