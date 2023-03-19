from __future__ import print_function, division
from abc import ABC
import torch
import torch.nn as nn


class ConvLayer(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ChannelAttention(nn.Module, ABC):
    def __init__(self, in_channels, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(in_channels * 2, in_channels * 2 * ratio)
        self.fc2 = nn.Linear(in_channels * 2 * ratio, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.squeeze(self.avg_pool(x))
        x_max = torch.squeeze(self.max_pool(x))
        if x.shape[0] == 1:
            x_avg = torch.unsqueeze(x_avg, dim=0)
            x_max = torch.unsqueeze(x_max, dim=0)
        x = torch.cat((x_avg, x_max), dim=1)
        x = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x


class SpatialAttention(nn.Module, ABC):
    def __init__(self, kernel_size=7, ratio=2):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(2, 2 * ratio, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv3d(2 * ratio, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_channels = x.shape[1]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        # x_tmp = x
        # for i in range(in_channels - 1):
        #     x = torch.cat((x, x_tmp), dim=1)
        return x


class Attention(nn.Module, ABC):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca = self.ca(x)
        for i in range(0, 3):
            ca = torch.unsqueeze(ca, dim=-1)
        x = torch.mul(x, ca)
        sa = self.sa(x)
        refined_x = torch.mul(x, sa)

        return refined_x

class _CNN_classification(nn.Module, ABC):
    def __init__(self, fil_num, drop_rate):
        super(_CNN_classification, self).__init__()

        self.block_c_1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block_a_1 = Attention(fil_num)
        self.block_c_2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (5, 1, 2), (3, 2, 0))
        self.block_a_2 = Attention(2 * fil_num)
        self.block_c_3 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.block_a_3 = Attention(2 * fil_num)
        self.block_c_4 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (3, 1, 0))
        self.block_a_4 = Attention(2 * fil_num)
        self.block_c_5 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.block_a_5 = Attention(2 * fil_num)
        self.block_c_6 = ConvLayer(2 * fil_num, 2 * fil_num, 0.1, (3, 1, 1), (1, 1, 0))
        self.block_a_6 = Attention(2 * fil_num)

        self.dense_c = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(32 * fil_num, 32)
        )
        self.classify = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2), )

    def forward(self, x):
        x_c = self.block_c_1(x)
        x_c = self.block_a_1(x_c)
        x_c = self.block_c_2(x_c)
        x_c = self.block_a_2(x_c)
        x_c = self.block_c_3(x_c)
        x_c = self.block_a_3(x_c)
        x_c = self.block_c_4(x_c)
        x_c = self.block_a_4(x_c)
        x_c = self.block_c_5(x_c)
        x_c = self.block_a_5(x_c)
        x_c = self.block_c_6(x_c)
        x_c = self.block_a_6(x_c)
        batch_size = x.shape[0]
        x_c = x_c.view(batch_size, -1)
        x_c = self.dense_c(x_c)
        output_c = self.classify(x_c)
        return output_c