"""
 Implementation based on MediaPipe and SSD: Single Shot MultiBox Detector:
 - https://arxiv.org/abs/2006.10214
 - https://arxiv.org/abs/1512.02325

"""

# To do:
# - Generate anchoring.
# Possible improvements:
# - FPN (Feature Pyramid Network - https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)

import torch
import torch.nn as nn

class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.first_stage = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False), # 256 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.second_stage = nn.Sequential(  
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 64 → 32
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True)
        )

        self.third_stage = nn.Sequential(  # 32 → 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True)
        )

        self.fourth_stage = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        X1 = self.first_stage(X)
        X2 = self.second_stage(X1)
        X3 = self.third_stage(X2)
        X4 = self.fourth_stage(X3)
        return [X1, X2, X3, X4]

class UpSample(nn.module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # The same way I was downsampling the input feature map by 2, I'm now upsampling it.
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels)

    def forward(self, X1, X2):
        X1 = self.up(X1)
        X = torch.cat([X1, X2], dim=1)
        return self.conv(X)

class AnchorPrediction(nn.Module):
    def __init__(self, in_channels, num_anchors=3, num_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # num_anchors = how many anchor boxes per pixel location
        self.num_anchors = num_anchors
        # 4 numbers for the bounding box | [cx, cy, w, h] or [x1, y1, x2, y2]
        # self.num_classes if it is palm or not
        # Each anchor has to predict 4 bounding box coordinares [cx, cy, w, h] and 1 class probability, so the output is 5.
        self.pred = nn.Conv2d(self.in_channels, self.num_anchors * (4 + self.num_classes), kernel_size=3, padding=1)

    def forward(self, X):
        B, _, _, _ = X.shape
        out = self.pred(X)  # Output shape is [B, A*(4+C), H, W]
        # B -> Batch size
        # Grouping predictions per spatial location
        # For each pixel at (h, w), we get A × (4 + C) values
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, A*(4+C)]
        out = out.view(B, -1, 4 + self.num_classes)  # [B, H*W*A, 4+C]
        return out
    
class SSDPalmDetector(nn.Module):
    def __init__(self, anchors: list = [6, 2, 2, 2]):
        super().__init__()
        self.down_convolutions = DownSample(in_channels=3)
        
        self.up_convolution_1 = UpSample(in_channels=512, out_channels=256)
        self.up_convolution_2 = UpSample(in_channels=256, out_channels=128)
        self.up_convolution_3 = UpSample(in_channels=128, out_channels=64)
        
        self.heads = nn.ModuleList([
            AnchorPrediction(64, anchors[0], 1),  
            AnchorPrediction(128, anchors[1], 1),
            AnchorPrediction(256, anchors[2], 1),
            AnchorPrediction(512, anchors[3], 1),
        ])
    
    def forward(self, x):
        # [X1(64ch), X2(128ch), X3(256ch), X4(512ch)]
        [X1, X2, X3, X4] = self.down_convolutions(x)
        
        up3 = self.up_convolution_1(X4, X3)   # 512->256, combine with X3
        up2 = self.up_convolution_2(up3, X2)  # 256->128, combine with X2  
        up1 = self.up_convolution_3(up2, X1)  # 128->64, combine with X1
        
        pyramid_features = [up1, up2, up3, X4]
        outputs = [head(fmap) for head, fmap in zip(self.heads, pyramid_features)]
        return torch.cat(outputs, dim=1)  # [B, N_anchors_total, 4 + C]
