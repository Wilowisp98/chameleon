"""
 Implementation based on:
 - https://youtu.be/IHq1t7NxS8k?si=d9dofGF9n96192R8
 - https://youtu.be/HS3Q_90hnDg?si=6BFVv_jLfQLhuA5i
 - https://d2l.ai/chapter_convolutional-modern/batch-norm.html
 - https://www.youtube.com/watch?v=oLvmLJkmXuc
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Kernel Size = 3 because I'm going to work with RGB images, so, there are going to be 3 channels.
        # What this means is that the matrix that represents the image is a matrix of weight * height * #channels size.
        # --
        # Stride = 1 so it moves to the "pixel" right after.
        # --
        # Padding = 1 so our matrix stays the same size after the convolution.
        # --
        # Batch Normalization: https://d2l.ai/chapter_convolutional-modern/batch-norm.html
        # Since batch normalization already includes bias, it makes no sense to have it on the convolution. (bias=False)
        # --
        # Since I don't need the previous convolution I can use ReLU inplace, it's faster and more memory efficient since:
        #   1. It doesn't create a new Tensor for the result.
        #   2. Since I'm not creating a new tensor => I don't need to allocate more memory.
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.dconv(X)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        # The kernel moves across the input feature map in 2x2 blocks.
        # By having a stride of 2 I'm ensuring that there is no overlap between the windows and that
        #   we are reducing the input size by half.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        down_X = self.conv(X)
        pooled_X = self.pool(down_X)
        return down_X, pooled_X
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The same way I was downsampling the input feature map by 2, I'm now upsampling it.
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, X1, X2):
        X1 = self.up(X1)
        # This is the skip connection mentioned in the UNET paper.
        # What we are doing is getting back the fine details that were lost by convoluting.
        X = torch.cat([X1, X2], dim=1)
        return self.conv(X)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # \
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        # _
        self.bottle_neck = DoubleConv(512, 1024)

        # /
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        # result
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, X):
        down_1, p1 = self.down_convolution_1(X)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        return self.final_conv(up_4)