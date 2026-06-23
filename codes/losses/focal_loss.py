"""Focal Loss 实现 - 用于处理正负样本不平衡"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: 聚焦参数，默认 2.0
        alpha: 平衡参数，默认 0.25
        reduction: 'mean' 或 'sum'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 确保 target 是 1D int64
        target = target.long().flatten()

        # 计算概率
        prob = F.softmax(pred, dim=1)

        # 取目标类别的概率
        p_t = prob[torch.arange(pred.size(0)), target]

        # 计算 Focal Loss
        focal_weight = (1 - p_t) ** self.gamma

        # α 平衡
        if self.alpha is not None:
            alpha_t = torch.full_like(p_t, self.alpha)
            alpha_t = torch.where(target == 1, alpha_t, 1 - alpha_t)
            focal_weight = alpha_t * focal_weight

        # 计算交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss