from torch import nn


class MainGan(nn.Module):
    def __init__(self, opt, data_depth, coder, critic, device, resolution, encoder_blocks, decoder_blocks, base_channels,
                 inverse_bottleneck, dropout):
        super(MainGan, self).__init__()
        self.encoder = coder(opt, is_encoder=True, data_depth=data_depth, encoder_blocks=encoder_blocks,
                             decoder_blocks=decoder_blocks,
                             base_channels=base_channels, device=device, inverse_bottleneck=inverse_bottleneck,
                             dropout=dropout)
        self.decoder = coder(opt, is_encoder=False, data_depth=data_depth, encoder_blocks=encoder_blocks,
                             decoder_blocks=decoder_blocks,
                             base_channels=base_channels, device=device, inverse_bottleneck=inverse_bottleneck,
                             dropout=dropout)
        # self.encoder = coder(data_depth, patch_size=1, is_encoder=True, resolution=resolution)
        # self.decoder = coder(data_depth, patch_size=1, is_encoder=False, resolution=resolution)
        self.critic = critic()
        self.to_device(device)

    def to_device(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.critic.to(device)

    def critic_params(self):
        return self.critic.parameters()

    def coder_params(self):
        return list(self.decoder.parameters()) + list(self.encoder.parameters())
