import torch
from torch import nn
import numpy as np


class RandomMask(nn.Module):
    def __init__(self, rate=((0.2, 0.4), (0.2, 0.4))):
        super(RandomMask, self).__init__()
        self.rate = rate

    def forward(self, x):
        num, _, h, w = x.shape
        for i in range(num):
            mask_h = int(h * (np.random.rand() * (self.rate[0][1] - self.rate[0][0]) + self.rate[0][0]))
            mask_w = int(w * (np.random.rand() * (self.rate[1][1] - self.rate[1][0]) + self.rate[1][0]))
            mask_id = np.random.randint(num)
            while mask_id == i:
                mask_id = np.random.randint(num)

            h_start = 0 if mask_h == h else np.random.randint(0, h - mask_h)
            w_start = 0 if mask_w == w else np.random.randint(0, w - mask_w)
            h_end = h_start + mask_h
            w_end = w_start + mask_w

            mask = torch.zeros_like(x[i])
            mask[:, h_start: h_end, w_start: w_end] = 1

            x[i] = x[mask_id] * mask + x[i] * (1 - mask)
            
        return x
            