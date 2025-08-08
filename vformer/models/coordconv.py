import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input_tensor):
        n, c, h, w = input_tensor.shape
        x = (2. / w * torch.arange(w, dtype=torch.float32, device=self.device) - 1).view(1, -1)
        y = (2. / h * torch.arange(h, dtype=torch.float32, device=self.device) - 1).view(1, -1)
        rangex = (torch.ones_like(y).T * x)[None, None, :, :].repeat(n, 1, 1, 1)
        rangey = (torch.ones_like(x).T * y).T[None, None, :, :].repeat(n, 1, 1, 1)

        return torch.cat([input_tensor, rangex, rangey], dim=1)


class CoordConv2d(conv.Conv2d):
    def __init__(self, device, in_channels, out_channels, kernel_size, padding=1, stride=1, dilation=1, groups=1,
                 bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.addcoords = AddCoords(device)
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
