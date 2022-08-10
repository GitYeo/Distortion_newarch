import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1,padding=1,spectral=False):
        super(ConvBNRelu, self).__init__()

        if spectral is False:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=padding),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                spectral_norm(nn.Conv2d(channels_in, channels_out, 3, stride, padding=padding)),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.layers(x)
