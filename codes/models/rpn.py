"""区域提议网络 (RPN)"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
import numpy as np
from typing import Dict, List, Tuple

from config import MODEL_CONFIG
from .attention import AttentionFactory


class RegionProposalNetwork3D(nn.Module):
    """3D区域提议网络"""

    def __init__(self, config: Dict = None):
        super().__init__()

        self.config = config or MODEL_CONFIG["rpn"]

        self.anchor_sizes = self.config.get("anchor_sizes", [16, 32, 64])
        self.anchor_ratios = self.config.get("anchor_ratios", [0.5, 1.0, 2.0])
        self.num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)

        self._create_backbone()
        self._create_rpn_heads()
        self._initialize_weights()

    def _create_backbone(self):
        backbone_config = MODEL_CONFIG["monai"]["unet3d"]

        self.backbone = UNet(
            spatial_dims=backbone_config["spatial_dims"],
            in_channels=backbone_config["in_channels"],
            out_channels=backbone_config["out_channels"],
            channels=backbone_config["channels"],
            strides=backbone_config["strides"],
            num_res_units=backbone_config["num_res_units"],
        )

        self.backbone_out_channels = backbone_config["channels"][-1]

        if self.config.get("attention", False):
            attention_type = self.config.get("attention_type", "se")
            self.attention = AttentionFactory.create_attention(
                attention_type=attention_type,
                spatial_dims=3,
                channels=self.backbone_out_channels,
                reduction_ratio=16
            )
        else:
            self.attention = None

    def _create_rpn_heads(self):
        self.rpn_cls = nn.Sequential(
            nn.Conv3d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, self.num_anchors * 2, kernel_size=1)
        )

        self.rpn_reg = nn.Sequential(
            nn.Conv3d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, self.num_anchors * 6, kernel_size=1)
        )

        dropout_rate = self.config.get("dropout_rate", 0.2)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]

        if self.attention is not None:
            features = self.attention(features)
        if self.dropout is not None:
            features = self.dropout(features)

        cls_logits = self.rpn_cls(features)
        reg_output = self.rpn_reg(features)

        batch_size = x.size(0)
        spatial_shape = cls_logits.shape[2:]

        cls_logits = cls_logits.view(batch_size, 2, self.num_anchors, *spatial_shape)
        reg_output = reg_output.view(batch_size, 6, self.num_anchors, *spatial_shape)

        anchors = self.generate_anchors(spatial_shape, x.device)

        return {
            'cls_logits': cls_logits,
            'reg_output': reg_output,
            'anchors': anchors,
            'features': features
        }

    def generate_anchors(self, feature_shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        D, H, W = feature_shape

        d_indices = np.arange(D)
        h_indices = np.arange(H)
        w_indices = np.arange(W)

        d_grid, h_grid, w_grid = np.meshgrid(d_indices, h_indices, w_indices, indexing='ij')

        centers_d = d_grid.flatten()
        centers_h = h_grid.flatten()
        centers_w = w_grid.flatten()

        all_anchors = []

        for size in self.anchor_sizes:
            for ratio in self.anchor_ratios:
                depth = size
                height = size
                width = size

                if ratio < 1.0:
                    depth *= ratio
                elif ratio > 1.0:
                    height *= ratio

                anchors = np.stack([
                    centers_d, centers_h, centers_w,
                    np.full_like(centers_d, depth),
                    np.full_like(centers_h, height),
                    np.full_like(centers_w, width)
                ], axis=1)

                all_anchors.append(anchors)

        anchors_np = np.concatenate(all_anchors, axis=0)
        anchors = torch.tensor(anchors_np, dtype=torch.float32, device=device)

        return anchors

    def decode_boxes(self, reg_output: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        batch_size = reg_output.size(0)
        num_anchors = anchors.size(0)

        reg_output = reg_output.permute(0, 2, 3, 4, 5, 1).contiguous()
        reg_output = reg_output.view(batch_size, num_anchors, 6)

        boxes = []
        for b in range(batch_size):
            reg = reg_output[b]
            decoded_boxes = torch.zeros_like(reg)
            decoded_boxes[:, :3] = anchors[:, :3] + reg[:, :3] * anchors[:, 3:]
            decoded_boxes[:, 3:] = anchors[:, 3:] * torch.exp(reg[:, 3:])
            boxes.append(decoded_boxes)

        boxes = torch.stack(boxes, dim=0)
        return boxes


class AnchorGenerator:
    def __init__(self, anchor_sizes: List[int], anchor_ratios: List[float]):
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios

    def generate_for_feature_map(self, feature_map_shape: Tuple[int, int, int],
                                  stride: int = 1) -> torch.Tensor:
        anchors = []

        for d in range(feature_map_shape[0]):
            for h in range(feature_map_shape[1]):
                for w in range(feature_map_shape[2]):
                    center_d = (d + 0.5) * stride
                    center_h = (h + 0.5) * stride
                    center_w = (w + 0.5) * stride

                    for size in self.anchor_sizes:
                        for ratio in self.anchor_ratios:
                            depth = size
                            height = size
                            width = size

                            if ratio < 1.0:
                                depth *= ratio
                            elif ratio > 1.0:
                                height *= ratio

                            anchor = [center_d, center_h, center_w, depth, height, width]
                            anchors.append(anchor)

        return torch.tensor(anchors, dtype=torch.float32)