"""结节分类器模型 - 3D CNN分类器"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import MODEL_CONFIG
from .attention import AttentionFactory
from codes.losses import FocalLoss

class NoduleClassifier(nn.Module):
    """肺结节分类器"""

    def __init__(self, config: Dict = None):
        super().__init__()

        self.config = config or MODEL_CONFIG["classifier"]

        self.in_channels = self.config.get("input_channels", 1)
        self.num_classes = self.config.get("num_classes", 2)

        # ========== 损失函数 ==========
        self.cls_loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean')

        # ========== 骨干网络 ==========
        self.backbone = nn.Sequential(
            # Block 1: 64³ → 32³
            nn.Conv3d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2: 32³ → 16³
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3: 16³ → 8³
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 4: 8³ → 4³
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 5: 4³ → 2³
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Global Pooling: 2³ → 1³
            nn.AdaptiveAvgPool3d(1),
        )
        self.backbone_out_channels = 512

        # ========== 注意力机制 ==========
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

        # ========== 分类头 ==========
        dropout_rate = self.config.get("dropout_rate", 0.3)
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.backbone_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, self.num_classes)
        )

        # ========== 回归头（边界框精修） ==========
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.backbone(x)

        if self.attention is not None:
            features = self.attention(features)

        cls_logits = self.classifier_head(features)
        reg_output = self.regression_head(features)

        return {
            'cls_logits': cls_logits,
            'reg_output': reg_output,
            'features': features
        }

    def compute_loss(self, cls_logits: torch.Tensor, reg_output: torch.Tensor,
                    labels: torch.Tensor, gt_boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失（Focal Loss + Smooth L1 Loss）"""
        # 分类损失：Focal Loss
        cls_loss = self.cls_loss_fn(cls_logits, labels)

        # 回归损失：Smooth L1 Loss（只对正样本）
        pos_mask = labels == 1
        if pos_mask.sum() > 0:
            reg_loss = F.smooth_l1_loss(reg_output[pos_mask], gt_boxes[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=cls_logits.device)

        total_loss = cls_loss + reg_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }

    def predict(self, x: torch.Tensor, proposals: torch.Tensor,
                confidence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """预测"""
        self.eval()

        with torch.no_grad():
            outputs = self.forward(x)
            cls_logits = outputs['cls_logits']
            reg_output = outputs['reg_output']

            cls_probs = F.softmax(cls_logits, dim=1)
            confidences, pred_labels = torch.max(cls_probs, dim=1)

            keep_mask = confidences >= confidence_threshold
            pred_labels = pred_labels[keep_mask]
            confidences = confidences[keep_mask]
            reg_output = reg_output[keep_mask]

            if proposals is not None and keep_mask.sum() > 0:
                pos_proposals = proposals[keep_mask]
                refined_boxes = self._refine_boxes(pos_proposals, reg_output)
            else:
                refined_boxes = None

            return {
                'labels': pred_labels,
                'confidences': confidences,
                'boxes': refined_boxes,
                'num_detections': keep_mask.sum()
            }

    def _refine_boxes(self, proposals: torch.Tensor, reg_output: torch.Tensor) -> torch.Tensor:
        """精修边界框"""
        refined_boxes = proposals.clone()
        refined_boxes[:, :3] += reg_output[:, :3]
        refined_boxes[:, 3:] *= torch.exp(reg_output[:, 3:])
        return refined_boxes