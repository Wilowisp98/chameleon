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

class tinySSD(nn.Module):
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 32 -> 16
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        X1 = self.first_stage(X)
        X2 = self.second_stage(X1)
        X3 = self.third_stage(X2)
        return [X1, X2, X3]

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
    def __init__(self, anchors: list = [2, 3, 4]):
        super().__init__()
        self.backbone = tinySSD(in_channels=3)
        self.heads = nn.ModuleList([
            AnchorPrediction(64, anchors[0], 1),
            AnchorPrediction(128, anchors[1], 1),
            AnchorPrediction(256, anchors[2], 1),
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        outputs = [head(fmap) for head, fmap in zip(self.heads, features)]
        return torch.cat(outputs, dim=1)  # [B, N_anchors_total, 4 + C]
