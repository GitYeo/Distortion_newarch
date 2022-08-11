import numpy as np
import torch
from torch import nn
from model.random_mask import RandomMask


class RandomPooling2D(nn.Module):
    def __init__(self, kernel=(2, 2), stride=2):
        super(RandomPooling2D, self).__init__()
        self.stride = stride
        self.kernel = kernel
        self.w_height = kernel[0]
        self.w_width = kernel[1]
        self.mask = RandomMask()

    def forward(self, x, mask=False):
        self.x = x
        if mask:
            self.x = self.mask(x)
        self.batch_size = x.shape[0]
        self.channels = x.shape[1]
        self.in_height = x.shape[2]
        self.in_width = x.shape[3]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        out = torch.zeros((self.batch_size, self.channels, self.out_height, self.out_width))

        pick_i = np.random.randint(self.w_height)
        pick_j = np.random.randint(self.w_width)
        
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                out[:, :, i, j] = self.x[:, :, start_i + pick_i, start_j + pick_j]
        return out.to(torch.device('cuda:0'))

    def backward(self, d_loss):
        dx = torch.zeros_like(self.x)

        for n in range(self.batch_size):
            for c in range(self.channels):
                for i in range(self.out_height):
                    for j in range(self.out_width):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.w_height
                        end_j = start_j + self.w_width
                        dx[n, c, start_i: end_i, start_j: end_j] = d_loss[n, c, i, j] / (self.w_width * self.w_height)
        return dx.to(torch.device('cuda:0'))
