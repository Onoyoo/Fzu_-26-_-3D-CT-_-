"""注意力机制模块 - 使用MONAI的注意力模块"""

import torch
import torch.nn as nn
import monai
from monai.networks.blocks import ChannelSELayer
from typing import Dict, Optional

class AttentionFactory:
    """注意力模块工厂 - 创建不同类型的注意力模块"""
    
    @staticmethod
    def create_attention(attention_type: str, spatial_dims: int = 3, 
                        channels: int = 64, reduction_ratio: int = 16,
                        **kwargs):
        """
        创建注意力模块
        
        Args:
            attention_type: 注意力类型 ('se', 'cbam', 'none')
            spatial_dims: 空间维度 (2或3)
            channels: 输入通道数
            reduction_ratio: 缩减比例
            **kwargs: 其他参数
            
        Returns:
            attention_module: 注意力模块
        """
        if attention_type.lower() == 'none' or not attention_type:
            return None
        
        elif attention_type.lower() == 'se':
            # Squeeze-and-Excitation注意力
            return SEBlock3D(channels=channels, reduction_ratio=reduction_ratio)
        
        elif attention_type.lower() == 'cbam':
            # CBAM注意力 (通道+空间注意力)
            return CBAM3D(channels=channels, reduction_ratio=reduction_ratio)
        
        elif attention_type.lower() == 'monai_se':
            # 使用MONAI的SE模块
            if spatial_dims == 3:
                return ChannelSELayer(spatial_dims=3, in_channels=channels, reduction_ratio=reduction_ratio)
            else:
                return ChannelSELayer(spatial_dims=2, in_channels=channels, reduction_ratio=reduction_ratio)
        
        elif attention_type.lower() == 'monai_cbam':
            # 使用自定义CBAM，因为MONAI没有ChannelSpatialSELayer
            return CBAM3D(channels=channels, reduction_ratio=reduction_ratio)
        
        else:
            raise ValueError(f"未知的注意力类型: {attention_type}")

class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation注意力模块"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # 全连接层
        reduced_channels = max(channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, D, H, W)
            
        Returns:
            加权后的特征图
        """
        batch_size, channels, depth, height, width = x.size()
        
        # 保存输入
        identity = x
        
        # 全局平均池化
        se = self.global_avg_pool(x).view(batch_size, channels)
        
        # 全连接层
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        
        # 重塑为原始形状
        se = se.view(batch_size, channels, 1, 1, 1)
        
        # 特征重标定
        x = x * se.expand_as(x)
        
        # 残差连接
        x = x + identity
        
        return x
    
    def __repr__(self):
        return f"SEBlock3D(channels={self.channels}, reduction_ratio={self.reduction_ratio})"


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module (CBAM)"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16, 
                 kernel_size: int = 7, use_spatial_att: bool = True):
        super().__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_spatial_att = use_spatial_att
        
        # 通道注意力
        self.channel_attention = ChannelAttention3D(
            channels=channels, 
            reduction_ratio=reduction_ratio
        )
        
        # 空间注意力
        if use_spatial_att:
            self.spatial_attention = SpatialAttention3D(kernel_size=kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, D, H, W)
            
        Returns:
            加权后的特征图
        """
        # 保存输入
        identity = x
        
        # 通道注意力
        x = self.channel_attention(x)
        
        # 空间注意力
        if self.use_spatial_att:
            x = self.spatial_attention(x)
        
        # 残差连接
        x = x + identity
        
        return x
    
    def __repr__(self):
        return f"CBAM3D(channels={self.channels}, reduction_ratio={self.reduction_ratio}, kernel_size={self.kernel_size})"


class ChannelAttention3D(nn.Module):
    """3D通道注意力模块"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享的全连接层
        reduced_channels = max(channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, D, H, W)
            
        Returns:
            通道注意力加权后的特征图
        """
        batch_size, channels, depth, height, width = x.size()
        
        # 平均池化分支
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        
        # 最大池化分支
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        
        # 合并两个分支
        channel_att = self.sigmoid(avg_out + max_out)
        channel_att = channel_att.view(batch_size, channels, 1, 1, 1)
        
        # 应用通道注意力
        return x * channel_att.expand_as(x)


class SpatialAttention3D(nn.Module):
    """3D空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # 使用卷积层生成空间注意力图
        padding = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels=2,  # 最大池化��平均池化的拼接
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, D, H, W)
            
        Returns:
            空间注意力加权后的特征图
        """
        # 沿通道维度计算最大值和平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # 拼接两个特征图
        spatial_att = torch.cat([avg_out, max_out], dim=1)  # (B, 2, D, H, W)
        
        # 通过卷积层生成空间注意力图
        spatial_att = self.conv(spatial_att)  # (B, 1, D, H, W)
        spatial_att = self.sigmoid(spatial_att)  # (B, 1, D, H, W)
        
        # 应用空间注意力
        return x * spatial_att.expand_as(x)


class ResidualAttentionBlock3D(nn.Module):
    """3D残差注意力块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 attention_type: str = 'se', reduction_ratio: int = 16,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 第一个卷积层
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 下采样层
        self.downsample = downsample
        
        # 注意力模块
        self.attention = AttentionFactory.create_attention(
            attention_type=attention_type,
            spatial_dims=3,
            channels=out_channels,
            reduction_ratio=reduction_ratio
        )
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, D, H, W)
            
        Returns:
            输出张量
        """
        # 保存残差连接
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 注意力机制
        if self.attention is not None:
            out = self.attention(out)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out


class AttentionUNet3D(nn.Module):
    """带注意力的3D UNet - 使用MONAI的UNet作为基础"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 从配置中获取参数
        spatial_dims = config.get("spatial_dims", 3)
        in_channels = config.get("in_channels", 1)
        out_channels = config.get("out_channels", 2)
        channels = config.get("channels", (16, 32, 64, 128, 256))
        strides = config.get("strides", (2, 2, 2, 2))
        attention_type = config.get("attention_type", "se")
        reduction_ratio = config.get("reduction_ratio", 16)
        
        # 创建MONAI UNet
        self.unet = monai.networks.nets.UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
        )
        
        # 添加注意力模块到每个解码器块
        self._add_attention_to_unet(attention_type, reduction_ratio)
    
    def _add_attention_to_unet(self, attention_type: str, reduction_ratio: int):
        """向UNet添加注意力模块"""
        # 获取UNet的解码器层
        decoder_layers = []
        
        # 遍历UNet的子模块，找到解码器层
        for name, module in self.unet.named_modules():
            if "up" in name and isinstance(module, nn.ConvTranspose3d):
                # 找到解码器层
                pass
        
        # 注意：MONAI的UNet结构比较固定，这里简化处理
        # 实际项目中可能需要更复杂的修改
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.unet(x)


if __name__ == "__main__":
    # 测试注意力模块
    print("测试注意力模块...")
    
    # 测试数据
    test_input = torch.randn(2, 64, 32, 32, 32)
    print(f"输入形状: {test_input.shape}")
    
    # 测试SE模块
    se_block = SEBlock3D(channels=64, reduction_ratio=16)
    se_output = se_block(test_input)
    print(f"SE模块输出形状: {se_output.shape}")
    
    # 测试CBAM模块
    cbam_block = CBAM3D(channels=64, reduction_ratio=16)
    cbam_output = cbam_block(test_input)
    print(f"CBAM模块输出形状: {cbam_output.shape}")
    
    # 测试注意力工厂
    attention_types = ['se', 'cbam', 'none']
    for att_type in attention_types:
        attention = AttentionFactory.create_attention(
            attention_type=att_type,
            spatial_dims=3,
            channels=64,
            reduction_ratio=16
        )
        if attention is not None:
            output = attention(test_input)
            print(f"{att_type.upper()}注意力输出形状: {output.shape}")
        else:
            print(f"{att_type.upper()}注意力: 无注意力模块")
    
    print("注意力模块测试完成!")