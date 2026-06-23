"""LUNA16数据预处理模块"""
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import torch
import torchio as tio
from typing import List, Tuple, Optional
import monai

from config import DATA_CONFIG, RAW_DATA_DIR, CSV_DIR


class CTScanPreprocessor:
    """CT扫描预处理类 """
    
    def __init__(self):
        self.config = DATA_CONFIG
        self.voxel_spacing = self.config["voxel_spacing"]
        self.crop_size = self.config.get("rpn_crop_size", [96, 96, 96])
        self.clip_min = self.config["ct_clip_min"]
        self.clip_max = self.config["ct_clip_max"]
        # 初始化MONAI转换器
        self._init_monai_transforms()
    
    def _init_monai_transforms(self):
        """初始化MONAI转换器"""
        # 重采样转换
        self.resample_transform = monai.transforms.Spacingd(
            keys=["image"],
            pixdim=self.voxel_spacing,
            mode="bilinear"
        )
        
        # 归一化转换
        self.normalize_transform = monai.transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=self.clip_min,
            a_max=self.clip_max,
            b_min=0.0,
            b_max=1.0,
            clip=True
        )
        
        # 裁剪转换
        self.crop_transform = monai.transforms.CenterSpatialCropd(
            keys=["image"],
            roi_size=self.crop_size
        )
    
    def load_ct_scan(self, mhd_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载CT扫描数据
        
        Args:
            mhd_path: .mhd文件路径
            
        Returns:
            ct_array: CT数据数组 (z, y, x)
            spacing: 体素间距 (z, y, x) 毫米
            origin: 图像原点 (z, y, x) 毫米
            direction: 图像方向矩阵
        """
        # 读取CT数据
        ct_sitk = sitk.ReadImage(str(mhd_path))

        # 获取CT数据和元数据
        ct_array = sitk.GetArrayFromImage(ct_sitk)  # (z, y, x)
        spacing = ct_sitk.GetSpacing()  # (z, y, x) 毫米
        origin = ct_sitk.GetOrigin()  # (z, y, x) 毫米
        direction = ct_sitk.GetDirection()  # 方向矩阵

        # 转换为numpy数组
        spacing = np.array(spacing, dtype=np.float32)
        origin = np.array(origin, dtype=np.float32)
        direction = np.array(direction, dtype=np.float32)

        return ct_array, spacing, origin, direction
    
    def world_to_voxel(self, world_coord: np.ndarray, origin: np.ndarray, 
                      spacing: np.ndarray) -> np.ndarray:
        """
        将世界坐标（毫米）转换为体素坐标
        
        Args:
            world_coord: 世界坐标 (x, y, z) 毫米
            origin: 图像原点 (x, y, z) 毫米
            spacing: 体素间距 (x, y, z) 毫米
            
        Returns:
            voxel_coord: 体素坐标 (z, y, x)
        """
        voxel_coord = (world_coord - origin) / spacing
        
        # 转换为整数
        voxel_coord = np.round(voxel_coord).astype(int)
        
        # 转换为(z, y, x)顺序
        voxel_coord = np.array([voxel_coord[2], voxel_coord[1], voxel_coord[0]])
        
        return voxel_coord
    
    def resample_ct_monai(self, ct_array: np.ndarray) -> np.ndarray:
        """
        重采样CT数据到统一间距
        
        Args:
            ct_array: 原始CT数据 (z, y, x)

        Returns:
            resampled_array: 重采样后的CT数据
        """
        ct_tensor = torch.from_numpy(ct_array).float().unsqueeze(0)

        # 创建数据字典
        data_dict = {"image": ct_tensor}

        # 应用重采样
        resampled_dict = self.resample_transform(data_dict)
        resampled_array = resampled_dict["image"].squeeze(0).numpy()

        return resampled_array

    def normalize_ct_monai(self, ct_array: np.ndarray) -> np.ndarray:
        """
        使用MONAI标准化CT值

        Args:
            ct_array: CT数据数组

        Returns:
            normalized_array: 标准化后的CT数据
        """

        ct_tensor = torch.from_numpy(ct_array).float().unsqueeze(0)

        # 创建数据字典
        data_dict = {"image": ct_tensor}

        # 应用标准化
        normalized_dict = self.normalize_transform(data_dict)
        normalized_array = normalized_dict["image"].squeeze(0).numpy()

        return normalized_array

    def extract_subvolume(self, ct_array: np.ndarray, center: np.ndarray,
                         size: List[int]) -> np.ndarray:
        """
        提取CT子体积

        Args:
            ct_array: CT数据数组
            center: 中心点坐标 (z, y, x)
            size: 子体积大小 (z, y, x)

        Returns:
            subvolume: 提取的子体积
        """

        ct_tensor = torch.from_numpy(ct_array).float().unsqueeze(0)

        # 创建裁剪转换器
        crop_transform = monai.transforms.SpatialCropd(
            keys=["image"],
            roi_center=[int(c) for c in center],
            roi_size=size
        )

        # 创建数据字典
        data_dict = {"image": ct_tensor}

        # 应用裁剪
        cropped_dict = crop_transform(data_dict)
        subvolume = cropped_dict["image"].squeeze(0).numpy()

        return subvolume

    def create_torchio_subject(self, ct_array: np.ndarray, spacing: np.ndarray,
                               label: Optional[int] = None, bbox: Optional[np.ndarray] = None):
        """
        创建TorchIO Subject用于数据增强

        Args:
            ct_array: CT数据数组
            spacing: 体素间距
            label: 标签（可选）
            bbox: 边界框（可选）

        Returns:
            subject: TorchIO Subject对象
        """
        # 转换为TorchIO Image
        ct_tensor = torch.from_numpy(ct_array).float()

        # 创建Subject
        subject_dict = {
            "ct": tio.ScalarImage(tensor=ct_tensor.unsqueeze(0)),
        }

        # 添加标签
        if label is not None:
            label_tensor = torch.full((1, *ct_array.shape), label, dtype=torch.long)
            subject_dict["label"] = tio.LabelMap(tensor=label_tensor)

        # 添加边界框
        if bbox is not None:
            bbox_tensor = torch.from_numpy(bbox).float()
            subject_dict["bbox"] = tio.LabelMap(tensor=bbox_tensor.unsqueeze(0))

        subject = tio.Subject(**subject_dict)

        # 设置间距
        subject["ct"].affine[0, 0] = spacing[0]
        subject["ct"].affine[1, 1] = spacing[1]
        subject["ct"].affine[2, 2] = spacing[2]

        return subject

class TorchioAugmentor:
    """TorchIO数据增强器"""

    def __init__(self):
        self.config = DATA_CONFIG.get("augmentation", {})

        # 初始化TorchIO转换器
        self._init_transforms()

    def _init_transforms(self):
        """初始化TorchIO转换器"""
        transforms = []

        # 随机翻转
        if self.config.get("flip_prob", 0) > 0:
            transforms.append(
                tio.RandomFlip(
                    axes=(0, 1, 2),
                    flip_probability=self.config["flip_prob"]
                )
            )

        # 随机旋转
        if self.config.get("rotation_range", 0) > 0:
            transforms.append(
                tio.RandomAffine(
                    scales=(1.0, 1.0),
                    degrees=self.config["rotation_range"],
                    translation=0,
                    center="image"
                )
            )

        # 随机缩放
        if self.config.get("scale_range", 0) > 0:
            scale_min = 1.0 - self.config["scale_range"]
            scale_max = 1.0 + self.config["scale_range"]
            transforms.append(
                tio.RandomAffine(
                    scales=(scale_min, scale_max),
                    degrees=0,
                    translation=0
                )
            )

        # 弹性形变
        if self.config.get("elastic_deformation", False):
            transforms.append(
                tio.RandomElasticDeformation(
                    num_control_points=7,
                    locked_borders=2
                )
            )

        # 随机偏置场
        if self.config.get("random_bias_field", False):
            transforms.append(
                tio.RandomBiasField(
                    coefficients=0.5
                )
            )

        if self.config.get("random_noise", False):
            transforms.append(
                tio.RandomNoise(
                    mean=0.0,
                    std=0.1
                )
            )

        # 强度变换
        if self.config.get("intensity_transform", False):
            transforms.append(
                tio.RandomBlur(std=(0, 2))  # 随机模糊
            )
            transforms.append(
                tio.RandomGamma(log_gamma=(-0.3, 0.3))  # Gamma校正
            )

        # 组合所有转换
        if transforms:
            self.transform = tio.Compose(transforms)
        else:
            self.transform = None

    def apply_augmentation(self, subject: tio.Subject) -> tio.Subject:
        """
        应用数据增强

        Args:
            subject: TorchIO Subject对象

        Returns:
            augmented_subject: 增强后的Subject
        """
        if self.transform is not None:
            return self.transform(subject)
        return subject

def load_annotations() -> pd.DataFrame:
    """
    加载标注数据

    Returns:
        annotations_df: 标注DataFrame
    """
    annotations_path = CSV_DIR / "annotations.csv"

    annotations_df = pd.read_csv(annotations_path)

    return annotations_df

def get_seriesuids() -> List[str]:
    """
    获取所有CT序列ID

    Returns:
        seriesuids: 序列ID列表
    """
    seriesuids_path = CSV_DIR / "seriesuids.csv"

    seriesuids_df = pd.read_csv(seriesuids_path)
    return seriesuids_df['seriesuid'].tolist()
