"""训练样本生成器"""

import random
from pathlib import Path
from typing import Dict, List, Optional
from config import DATA_CONFIG, PROCESSED_DIR, RAW_DATA_DIR, CSV_DIR  # ✅ 添加 CSV_DIR

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import argparse
from tqdm import tqdm

from config import  DATA_CONFIG, PROCESSED_DIR, RAW_DATA_DIR, RPN_MODEL_PATH
from .preprocess import CTScanPreprocessor, TorchioAugmentor, load_annotations
from codes.models.rpn import RegionProposalNetwork3D

class RPNSampleGenerator:
    """RPN训练样本生成器"""

    def __init__(self):
        self.config = DATA_CONFIG
        self.preprocessor = CTScanPreprocessor()
        self.augmentor = TorchioAugmentor()
        self.crop_size = self.config.get("rpn_crop_size", [96, 96, 96])

    def generate_samples_from_ct(self, ct_path: Path, annotations_df: pd.DataFrame,
                                  output_dir: Path) -> int:
        """从单个CT扫描生成RPN训练样本"""
        ct_data, spacing, origin, direction = self.preprocessor.load_ct_scan(ct_path)
        resampled_array = self.preprocessor.resample_ct_monai(ct_data)
        normalized_array = self.preprocessor.normalize_ct_monai(resampled_array)
        seriesuid = ct_path.stem

        ct_annotations = annotations_df[annotations_df['seriesuid'] == seriesuid]
        if ct_annotations.empty:
            return 0

        positive_samples = self._generate_positive_samples(
            normalized_array, ct_annotations, spacing, origin, output_dir, seriesuid
        )
        negative_samples = self._generate_negative_samples(
            normalized_array, ct_annotations, spacing, origin, output_dir, seriesuid
        )

        return positive_samples + negative_samples

    def _generate_positive_samples(self, ct_array: np.ndarray, annotations: pd.DataFrame,
                                   spacing: np.ndarray, origin: np.ndarray,
                                   output_dir: Path, seriesuid: str) -> int:
        """生成正样本（包含结节的区域）"""
        generated = 0
        crop_size = self.crop_size

        for idx, (_, row) in enumerate(annotations.iterrows()):
            world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
            diameter_mm = row['diameter_mm']

            voxel_coord = self.preprocessor.world_to_voxel(world_coord, origin, spacing)
            subvolume = self._extract_patch_with_padding(ct_array, voxel_coord, crop_size)

            if subvolume is None:
                continue

            bbox_center = np.array(crop_size) // 2
            diameter_voxel = diameter_mm / np.mean(spacing)
            bbox_size = np.array([diameter_voxel, diameter_voxel, diameter_voxel])

            sample_id = f"{seriesuid}_pos_{generated:04d}"
            self._save_rpn_sample(subvolume, bbox_center, bbox_size, 1, output_dir, sample_id)
            generated += 1

            if generated < 3:
                aug_count = self._generate_augmented_samples(
                    subvolume, bbox_center, bbox_size, output_dir, seriesuid, generated
                )
                generated += aug_count

        return generated

    def _extract_patch_with_padding(self, ct_array: np.ndarray, center: np.ndarray,
                                    patch_size: List[int]) -> Optional[np.ndarray]:
        """提取体素块，支持边界填充"""
        subvolume = self.preprocessor.extract_subvolume(ct_array, center, patch_size)

        if subvolume is None:
            return None

        # 如果是 2D，添加一个维度变成 3D
        if subvolume.ndim == 2:
            subvolume = subvolume[np.newaxis, ...]  # (H, W) → (1, H, W)

        if subvolume.shape != tuple(patch_size):
            padded_patch = np.zeros(patch_size, dtype=np.float32)
            offset_d = (patch_size[0] - subvolume.shape[0]) // 2
            offset_h = (patch_size[1] - subvolume.shape[1]) // 2
            offset_w = (patch_size[2] - subvolume.shape[2]) // 2
            padded_patch[offset_d:offset_d+subvolume.shape[0],
                        offset_h:offset_h+subvolume.shape[1],
                        offset_w:offset_w+subvolume.shape[2]] = subvolume
            return padded_patch

        return subvolume

    def _generate_negative_samples(self, ct_array: np.ndarray, annotations: pd.DataFrame,
                                   spacing: np.ndarray, origin: np.ndarray,
                                   output_dir: Path, seriesuid: str) -> int:
        """生成负样本（背景区域）"""
        generated = 0
        crop_size = self.crop_size
        ct_shape = ct_array.shape

        nodule_positions = []
        for _, row in annotations.iterrows():
            world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
            voxel_coord = self.preprocessor.world_to_voxel(world_coord, origin, spacing)
            nodule_positions.append(voxel_coord)

        num_neg_samples = min(len(nodule_positions) * 3, 15) if nodule_positions else 5
        generated_positions = set()

        for _ in range(num_neg_samples):
            valid_position = False
            attempts = 0

            while not valid_position and attempts < 100:
                random_pos = np.zeros(3, dtype=int)
                half_size = np.array(crop_size) // 2

                for j in range(3):
                    lower_bound = half_size[j]
                    upper_bound = ct_shape[j] - half_size[j]

                    if upper_bound <= lower_bound:
                        lower_bound = 0
                        upper_bound = max(0, ct_shape[j] - crop_size[j])

                    if upper_bound > lower_bound:
                        random_pos[j] = random.randint(lower_bound, upper_bound)
                    else:
                        random_pos[j] = max(0, min(ct_shape[j] // 2, ct_shape[j] - crop_size[j]))

                pos_key = tuple(random_pos)
                if pos_key in generated_positions:
                    attempts += 1
                    continue

                is_valid = True
                if nodule_positions:
                    for nodule_pos in nodule_positions:
                        distance_mm = np.linalg.norm((random_pos - nodule_pos) * np.mean(spacing))
                        if distance_mm < 30:
                            is_valid = False
                            break

                if is_valid:
                    valid_position = True
                    generated_positions.add(pos_key)

                attempts += 1

            if not valid_position:
                continue

            subvolume = self._extract_patch_with_padding(ct_array, random_pos, crop_size)
            if subvolume is None:
                continue

            sample_id = f"{seriesuid}_neg_{generated:04d}"
            self._save_rpn_sample(subvolume, np.zeros(3), np.zeros(3), 0, output_dir, sample_id)
            generated += 1

        return generated

    def _generate_augmented_samples(self, subvolume: np.ndarray, bbox_center: np.ndarray,
                                    bbox_size: np.ndarray, output_dir: Path,
                                    seriesuid: str, base_count: int) -> int:
        """生成增强样本（随机小角度旋转）"""
        generated = 0

        for aug_idx in range(2):
            angle = random.uniform(-30, 30)
            axis = random.randint(0, 2)
            axes = [(1, 2), (0, 2), (0, 1)]

            augmented = ndi.rotate(
                subvolume, angle, axes=axes[axis],
                reshape=False, order=1, mode='constant', cval=0.0
            )

            sample_id = f"{seriesuid}_aug_{base_count + aug_idx:04d}"
            self._save_rpn_sample(augmented, bbox_center, bbox_size, 1, output_dir, sample_id)
            generated += 1

        return generated

    def _save_rpn_sample(self, data: np.ndarray, bbox_center: np.ndarray,
                         bbox_size: np.ndarray, label: int, output_dir: Path,
                         sample_id: str):
        """保存RPN样本数据"""
        output_dir.mkdir(parents=True, exist_ok=True)

        data_path = output_dir / f"{sample_id}_data.npy"
        np.save(data_path, data.astype(np.float32))

        label_data = {
            'label': label,
            'bbox_center': bbox_center.astype(np.float32),
            'bbox_size': bbox_size.astype(np.float32),
            'has_bbox': label == 1
        }
        label_path = output_dir / f"{sample_id}_label.npy"
        np.save(label_path, label_data)

class ClassifierSampleGenerator:
    """分类器训练样本生成器 - 使用官方候选"""

    def __init__(self):
        self.config = DATA_CONFIG
        self.preprocessor = CTScanPreprocessor()
        self.augmentor = TorchioAugmentor()
        self.crop_size = self.config.get("classifier_crop_size", [64, 64, 64])
        # ✅ 不需要 RPN 模型了

    def generate_samples(self, ct_paths: List[Path], annotations_df: pd.DataFrame,
                         output_dir: Path, max_cts: Optional[int] = None) -> int:
        """生成分类器训练样本（使用官方候选）"""
        if max_cts is not None:
            ct_paths = ct_paths[:max_cts]
        total_generated = 0

        # ✅ 加载官方候选
        candidates_df = pd.read_csv(CSV_DIR / "candidates.csv")

        for ct_path in tqdm(ct_paths, desc="生成分类器样本"):
            ct_data, spacing, origin, direction = self.preprocessor.load_ct_scan(ct_path)
            if ct_data is None:
                continue

            resampled_array = self.preprocessor.resample_ct_monai(ct_data)
            normalized_array = self.preprocessor.normalize_ct_monai(resampled_array)

            seriesuid = ct_path.stem
            ct_annotations = annotations_df[annotations_df['seriesuid'] == seriesuid]
            ct_candidates = candidates_df[candidates_df['seriesuid'] == seriesuid]

            if ct_annotations.empty:
                continue

            # 获取真实结节位置
            true_nodules = []
            for _, row in ct_annotations.iterrows():
                world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
                voxel_coord = self.preprocessor.world_to_voxel(world_coord, origin, spacing)
                true_nodules.append(voxel_coord)

            positive_count = 0
            negative_count = 0

            for _, cand in ct_candidates.iterrows():
                # 候选点坐标
                cand_center = np.array([cand['coordX'], cand['coordY'], cand['coordZ']])
                cand_voxel = self.preprocessor.world_to_voxel(cand_center, origin, spacing)

                # 检查是否与真实结节重合
                is_positive = False
                for nodule in true_nodules:
                    distance_mm = np.linalg.norm(cand_voxel - nodule) * np.mean(spacing)
                    if distance_mm < 20:
                        is_positive = True
                        break

                # 裁剪 64³ 体素块
                subvolume = self._extract_patch_with_padding(normalized_array, cand_voxel.astype(int), self.crop_size)
                if subvolume is None:
                    continue

                label = 1 if is_positive else 0
                if label == 1:
                    pos_dir = output_dir / "positive"
                    sample_id = f"{seriesuid}_pos_{positive_count:04d}"
                    self._save_classifier_sample(subvolume, label, pos_dir, sample_id)
                    positive_count += 1
                else:
                    neg_dir = output_dir / "negative"
                    sample_id = f"{seriesuid}_neg_{negative_count:04d}"
                    self._save_classifier_sample(subvolume, label, neg_dir, sample_id)
                    negative_count += 1

            total_generated += positive_count + negative_count
            if positive_count + negative_count > 0:
                print(f"  {ct_path.name}: {positive_count}正样本 + {negative_count}负样本")

        return total_generated

    def _extract_patch_with_padding(self, ct_array: np.ndarray, center: np.ndarray,
                                    patch_size: tuple) -> Optional[np.ndarray]:
        """提取体素块，支持边界填充"""
        subvolume = self.preprocessor.extract_subvolume(ct_array, center, list(patch_size))

        if subvolume is None:
            return None

        if subvolume.ndim == 2:
            subvolume = subvolume[np.newaxis, ...]

        if subvolume.shape != patch_size:
            padded_patch = np.zeros(patch_size, dtype=np.float32)
            offset_d = (patch_size[0] - subvolume.shape[0]) // 2
            offset_h = (patch_size[1] - subvolume.shape[1]) // 2
            offset_w = (patch_size[2] - subvolume.shape[2]) // 2
            padded_patch[offset_d:offset_d+subvolume.shape[0],
                        offset_h:offset_h+subvolume.shape[1],
                        offset_w:offset_w+subvolume.shape[2]] = subvolume
            return padded_patch

        return subvolume

    def _save_classifier_sample(self, data: np.ndarray, label: int,
                                output_dir: Path, sample_id: str):
        """保存分类器样本"""
        output_dir.mkdir(parents=True, exist_ok=True)
    
        data_path = output_dir / f"{sample_id}_data.npy"
        
        if data_path.exists():
            return
    
        np.save(data_path, data.astype(np.float32))
    
        label_path = output_dir / f"{sample_id}_label.npy"
        np.save(label_path, np.array([label], dtype=np.int64))


# ==================== 顶层函数 ====================
def generate_rpn_samples(max_cts: Optional[int] = None) -> int:
    """生成RPN训练样本"""

    output_dir = PROCESSED_DIR / "rpn_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_df = load_annotations()
    ct_files = list(RAW_DATA_DIR.glob("**/*.mhd"))

    if not ct_files:
        print("未找到CT文件，请检查raw/目录")
        return 0

    ct_files = ct_files[:max_cts]

    generator = RPNSampleGenerator()
    total_samples = 0

    for ct_path in tqdm(ct_files, desc="处理CT文件"):
        samples = generator.generate_samples_from_ct(ct_path, annotations_df, output_dir)
        total_samples += samples

    data_files = list(output_dir.glob("*_data.npy"))
    positive_data = [f for f in data_files if "_pos_" in f.name]
    negative_data = [f for f in data_files if "_neg_" in f.name]
    aug_data = [f for f in data_files if "_aug_" in f.name]

    print(f"\nRPN样本生成完成!")
    print(f"总生成样本数: {total_samples}")
    print(f"正样本: {len(positive_data)}")
    print(f"负样本: {len(negative_data)}")
    print(f"增强样本: {len(aug_data)}")
    print(f"样本保存在: {output_dir}")

    return total_samples


def generate_classifier_samples(max_cts: Optional[int] = None, start_idx: int = 0) -> int:
    """生成分类器训练样本
    
    Args:
        max_cts: 最大处理的CT文件数量（默认: 全部）
        start_idx: 从第几个CT开始处理（默认: 0）
    """
    output_dir = PROCESSED_DIR / "classifier_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations_df = load_annotations()
    ct_files = list(RAW_DATA_DIR.glob("**/*.mhd"))

    if not ct_files:
        print("未找到CT文件，请检查raw/目录")
        return 0

    # ✅ 从 start_idx 开始取
    if max_cts is not None:
        ct_files = ct_files[start_idx:start_idx + max_cts]
    else:
        ct_files = ct_files[start_idx:]

    print(f"从第 {start_idx} 个CT开始，处理 {len(ct_files)} 个CT文件")

    generator = ClassifierSampleGenerator()
    total_samples = generator.generate_samples(ct_files, annotations_df, output_dir, max_cts)

    # 统计样本数量
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"

    pos_count = len(list(pos_dir.glob("*_data.npy"))) if pos_dir.exists() else 0
    neg_count = len(list(neg_dir.glob("*_data.npy"))) if neg_dir.exists() else 0

    print(f"\n分类器样本生成完成!")
    print(f"总生成样本数: {total_samples}")
    print(f"正样本: {pos_count}")
    print(f"负样本: {neg_count}")
    print(f"样本保存在: {output_dir}")

    return total_samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="生成训练样本")
    parser.add_argument("--mode", type=str, choices=["rpn", "classifier"], required=True,
                       help="生成模式")
    parser.add_argument("--max-cts", type=int, default=None,
                       help="最大处理的CT文件数量（默认: 全部）")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="从第几个CT开始处理（默认: 0）")

    args = parser.parse_args()

    if args.mode == "rpn":
        generate_rpn_samples(args.max_cts)
    elif args.mode == "classifier":
        generate_classifier_samples(args.max_cts, args.start_idx)