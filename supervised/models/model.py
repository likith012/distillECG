import torch.nn as nn
import torch

from .resnet1d import BaseNet


class attention(nn.Module):
    """
    Class for the attention module of the model

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the attention module

    """

    def __init__(self):
        super(attention, self).__init__()
        self.att_dim = 256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e * self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


class encoder(nn.Module):
    """
    Class for the encoder of the model that contains time encoder

    Methods:
    --------
        forward: torch.Tensor -> torch.Tensor
            forward pass of the encoder

    """

    def __init__(self, input_channels):
        super(encoder, self).__init__()
        self.base_module = BaseNet(input_channels=input_channels)
        self.attention_module = attention()

    def forward(self, x: torch.Tensor):
        x = self.base_module(x)
        x = self.attention_module(x)
        return x


class sleep_model(nn.Module):
    def __init__(self, input_channels, epoch_len):
        super(sleep_model, self).__init__()

        self.epoch_len = epoch_len
        self.base_encoder = encoder(input_channels)

    def forward(self, x, y):
        x = self.base_encoder(x[:, (self.epoch_len // 2), :, :])
        return x
