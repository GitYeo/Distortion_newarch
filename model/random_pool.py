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
        # self.x = x.cpu().data.numpy()
        self.x = x
        if mask:
            self.x = self.mask(x)
        self.batch_size = x.shape[0]
        self.channels = x.shape[1]
        self.in_height = x.shape[2]
        self.in_width = x.shape[3]

        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        # out = np.zeros((self.batch_size, self.channels, self.out_height, self.out_width))
        out = torch.zeros((self.batch_size, self.channels, self.out_height, self.out_width))

        pick_i = np.random.randint(self.w_height)
        pick_j = np.random.randint(self.w_width)
        # pick_i = torch.randint(self.w_height)
        # pick_j = torch.randint(self.w_width)

        # for n in range(self.batch_size):
        #     for c in range(self.channels):
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                # end_i = start_i + self.w_height
                # end_j = start_j + self.w_width
                # out[:, :, i, j] = np.mean(x[:, :, start_i: end_i, start_j: end_j], (2, 3))
                out[:, :, i, j] = self.x[:, :, start_i + pick_i, start_j + pick_j]
        # return torch.tensor(out, requires_grad=True, dtype=torch.float32, device='cuda:0')
        return out.to(torch.device('cuda:0'))

    def backward(self, d_loss):
        # dx = np.zeros_like(self.x)
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
        # return torch.tensor(dx, requires_grad=True, dtype=torch.float32, device='cuda:0')
        return dx.to(torch.device('cuda:0'))

# # test
# # np.set_printoptions(precision=8, suppress=True, linewidth=120)
# # x_numpy = np.random.random((2, 3, 8, 8))
# # x = torch.tensor(x_numpy, requires_grad=True)
# # yn = np.mean(x_numpy)
# # y = x.mean()
# # print(yn, y)
# # print(torch.tensor(yn) == y)
# x = torch.rand((2, 3, 8, 8), requires_grad=True)
# # official = torch.nn.AvgPool2d((2, 2), 2)
# my = RandomPooling2D((2, 2), stride=2)
# # out_off = official(x)
# out_my = my(x)
# # print(out_off.shape)
# print(out_my.shape)
# # print((out_off == out_my).all())
# d_loss_numpy = np.random.random(out_my.shape)
# # d_loss_tensor = torch.tensor(d_loss_numpy, requires_grad=True)
# # d_loss = torch.rand(out_off.shape, requires_grad=True)
# # out_off.backward(d_loss_tensor)
# dx_my = my.backward(d_loss_numpy)
# # dx_off = x.grad
# # print(dx_off.shape)
# print(dx_my.shape)
# # print((dx_off == dx_my).all())

# print('input \n', x)

# print("out_my \n", out_my)
# # print("out_off \n", out_off.data.numpy())

# print("dx_my \n", dx_my)
# # print("dx_off \n", dx_off.data.numpy())

# # print((out_off == torch.tensor(out_my)).all())
# # print((dx_off == torch.tensor(dx_my)).all())
