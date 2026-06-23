"""LUNA16数据集模块"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import torchio as tio
from typing import Dict, List, Tuple, Optional, Any
import random

from tqdm import tqdm

from config import DATA_CONFIG, PROCESSED_DIR
from .preprocess import CTScanPreprocessor, load_annotations

class BaseCTDataset(Dataset):
    """基础CT数据集类"""
    
    def __init__(self, config: Dict = None, transform: Optional[tio.Transform] = None):
        super().__init__()
        self.config = config or DATA_CONFIG
        self.transform = transform
        self.preprocessor = CTScanPreprocessor()
        
        # 数据集列表
        self.samples = []
        
        # 数据增强配置
        self.augmentation_config = self.config.get("augmentation", {})
        
        # 初始化数据增强转换
        self._init_transforms()

    def _init_transforms(self):
        """初始化数据增强转换"""
        transforms = [
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=self.augmentation_config["flip_prob"]),
            tio.RandomAffine(scales=(1.0, 1.0), degrees=self.augmentation_config["rotation_range"]),
            tio.RandomAffine(
                scales=(1.0 - self.augmentation_config["scale_range"],
                        1.0 + self.augmentation_config["scale_range"]),
                degrees=0
            ),
        ]

        if self.augmentation_config.get("elastic_deformation", False):
            transforms.append(tio.RandomElasticDeformation(num_control_points=7, locked_borders=2))

        if self.augmentation_config.get("random_bias_field", False):
            transforms.append(tio.RandomBiasField(coefficients=0.5))

        if self.augmentation_config.get("random_noise", False):
            transforms.append(tio.RandomNoise(mean=0.0, std=0.1))

        if self.augmentation_config.get("intensity_transform", False):
            transforms.append(tio.RandomBlur(std=(0, 2)))
            transforms.append(tio.RandomGamma(log_gamma=(-0.3, 0.3)))

        self.torchio_transform = tio.Compose(transforms) if transforms else None

    def apply_augmentation(self, image: torch.Tensor, label: Optional[torch.Tensor] = None,
                           bbox: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.torchio_transform is None:
            return {'image': image, 'label': label, 'bbox': bbox}

        subject_dict = {'image': tio.ScalarImage(tensor=image)}

        if label is not None:
            # 用 label 的值填满整个 4D 张量
            label_value = label.item() if torch.is_tensor(label) else label
            label_tensor = torch.full((1, *image.shape[1:]), label_value, dtype=torch.long)
            subject_dict['label'] = tio.LabelMap(tensor=label_tensor)

        if bbox is not None:
            bbox_tensor = torch.zeros((1, *image.shape[1:]), dtype=torch.float)
            subject_dict['bbox'] = tio.LabelMap(tensor=bbox_tensor)

        subject = tio.Subject(**subject_dict)
        augmented_subject = self.torchio_transform(subject)

        result = {'image': augmented_subject['image'].tensor}

        if 'label' in augmented_subject:
            result['label'] = augmented_subject['label'].tensor
        if 'bbox' in augmented_subject:
            result['bbox'] = augmented_subject['bbox'].tensor

        return result

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("子类必须实现此方法")

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """批处理函数 - 处理混合类型（张量、字符串等）"""
        if not batch:
            return {}

        collated = {}

        for key in batch[0].keys():
            values = [item.get(key) for item in batch if key in item]

            if not values:
                continue

            first_val = values[0]

            if isinstance(first_val, str):
                collated[key] = values
            elif isinstance(first_val, torch.Tensor):
                # 如果所有张量形状相同，stack；否则保持列表
                try:
                    collated[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    collated[key] = values
            elif isinstance(first_val, (list, tuple)):
                collated[key] = values
            elif isinstance(first_val, (int, float, bool)):
                collated[key] = torch.tensor(values)
            else:
                # 其他类型（如 None 或其他对象）直接保持列表
                collated[key] = values

        return collated

class RPNDataset(BaseCTDataset):
    """RPN训练数据集"""
    
    def __init__(self, data_dir: Path = None, mode: str = 'train', 
                 config: Dict = None, transform: Optional[tio.Transform] = None):
        super().__init__(config, transform)
        
        self.mode = mode
        self.data_dir = data_dir or PROCESSED_DIR / "rpn_samples"
        
        self._load_samples()
        self._split_dataset()
        
        print(f"RPN数据集: {self.mode}模式, {len(self.samples)}个样本")
    
    def _load_samples(self):
        """加载样本数据"""
        data_files = list(self.data_dir.glob("*_data.npy"))
        
        for data_file in data_files:
            label_file = self.data_dir / f"{data_file.stem.replace('_data', '_label')}.npy"
            
            if label_file.exists():
                self.samples.append({
                    'data_path': data_file,
                    'label_path': label_file,
                    'sample_id': data_file.stem.replace('_data', '')
                })
    
    def _split_dataset(self):
        """分割数据集"""
        if not self.samples:
            return
        
        random.seed(42)
        random.shuffle(self.samples)
        
        total_samples = len(self.samples)
        train_ratio = self.config.get("train_ratio", 0.7)
        val_ratio = self.config.get("val_ratio", 0.15)
        
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        if self.mode == 'train':
            self.samples = self.samples[:train_end]
        elif self.mode == 'val':
            self.samples = self.samples[train_end:val_end]
        elif self.mode == 'test':
            self.samples = self.samples[val_end:]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample_info = self.samples[idx]
        
        try:
            data = np.load(sample_info['data_path'])

            label_data = np.load(sample_info['label_path'], allow_pickle=True).item()
            
            image = torch.from_numpy(data).float().unsqueeze(0)
            label = torch.tensor([label_data['label']], dtype=torch.long)
            bbox_center = torch.from_numpy(label_data['bbox_center']).float()
            bbox_size = torch.from_numpy(label_data['bbox_size']).float()
            has_bbox = torch.tensor([label_data['has_bbox']], dtype=torch.bool)
            bbox = torch.cat([bbox_center, bbox_size])
            
            if self.mode == 'train' and self.torchio_transform is not None:
                augmented = self.apply_augmentation(image, label.unsqueeze(0), bbox.unsqueeze(0))
                image = augmented['image']
                label = augmented['label'].squeeze(0)
                bbox = augmented['bbox'].squeeze(0)
            
            return {
                'image': image,
                'label': label,
                'bbox': bbox,
                'has_bbox': has_bbox
            }
            
        except Exception as e:
            print(f"加载样本失败 {sample_info['sample_id']}: {e}")
            return self._create_empty_sample()
    
    def _create_empty_sample(self) -> Dict[str, torch.Tensor]:
        """创建空样本"""
        crop_size = self.config.get("rpn_crop_size", [96, 96, 96])
        return {
            'image': torch.zeros(1, *crop_size),
            'label': torch.tensor([0]),
            'bbox': torch.zeros(6),
            'has_bbox': torch.tensor([False])
        }

class ClassifierDataset(BaseCTDataset):
    """分类器训练数据集（正样本增强 + 数据增强）"""

    def __init__(self, data_dir: Path = None, mode: str = 'train',
                 config: Dict = None, transform: Optional[tio.Transform] = None):
        super().__init__(config, transform)

        self.mode = mode
        self.data_dir = data_dir or PROCESSED_DIR / "classifier_samples"
        self.cache = {}

        self._load_samples()
        self._split_dataset()
        self._preload_to_cache()

        print(f"分类器数据集: {self.mode}模式, {len(self.samples)}个样本 (已缓存)")

    def _preload_to_cache(self):
        """预加载所有样本到内存"""
        print(f"   预加载 {len(self.samples)} 个样本到内存...")
        for i, sample_info in enumerate(tqdm(self.samples, desc="   缓存样本")):
            try:
                data = np.load(sample_info['data_path']).astype(np.float32)
                label = np.load(sample_info['label_path'])

                bbox_path_str = str(sample_info['data_path']).replace('_data.npy', '_bbox.npy')
                if Path(bbox_path_str).exists():
                    bbox = np.load(bbox_path_str).astype(np.float32)
                else:
                    bbox = np.zeros(6, dtype=np.float32)

                if isinstance(label, np.ndarray):
                    label = int(label.item() if label.size == 1 else label[0])
                else:
                    label = int(label)

                self.cache[i] = {
                    'data': data,
                    'label': label,
                    'bbox': bbox
                }
            except Exception as e:
                self.cache[i] = None

        total_bytes = sum(
            s['data'].nbytes + s['bbox'].nbytes
            for s in self.cache.values() if s is not None
        )
        print(f"   缓存完成，内存占用约 {total_bytes / 1024**3:.1f} GB")

    def _load_samples(self):
        """加载样本路径"""
        pos_dir = self.data_dir / "positive"
        if pos_dir.exists():
            for data_file in pos_dir.glob("*_data.npy"):
                label_file = pos_dir / f"{data_file.stem.replace('_data', '_label')}.npy"
                if label_file.exists():
                    self.samples.append({
                        'data_path': data_file,
                        'label_path': label_file,
                        'label': 1,
                        'sample_id': data_file.stem.replace('_data', '')
                    })

        neg_dir = self.data_dir / "negative"
        if neg_dir.exists():
            for data_file in neg_dir.glob("*_data.npy"):
                label_file = neg_dir / f"{data_file.stem.replace('_data', '_label')}.npy"
                if label_file.exists():
                    self.samples.append({
                        'data_path': data_file,
                        'label_path': label_file,
                        'label': 0,
                        'sample_id': data_file.stem.replace('_data', '')
                    })

    def _split_dataset(self):
        """分割数据集（正样本×3，负样本 1/10）"""
        if not self.samples:
            return

        pos_samples = [s for s in self.samples if s['label'] == 1]
        neg_samples = [s for s in self.samples if s['label'] == 0]

        random.seed(42)
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)

        # ✅ 负样本只保留 1/10
        neg_samples = random.sample(neg_samples, len(neg_samples) // 10)

        train_ratio = self.config.get("train_ratio", 0.7)
        val_ratio = self.config.get("val_ratio", 0.15)

        pos_train_end = int(len(pos_samples) * train_ratio)
        pos_val_end = pos_train_end + int(len(pos_samples) * val_ratio)

        neg_train_end = int(len(neg_samples) * train_ratio)
        neg_val_end = neg_train_end + int(len(neg_samples) * val_ratio)

        if self.mode == 'train':
            train_pos = pos_samples[:pos_train_end]
            train_neg = neg_samples[:neg_train_end]
            # ✅ 正样本重复 3 倍
            self.samples = train_pos * 3 + train_neg
        elif self.mode == 'val':
            self.samples = pos_samples[pos_train_end:pos_val_end] + neg_samples[neg_train_end:neg_val_end]
        elif self.mode == 'test':
            self.samples = pos_samples[pos_val_end:] + neg_samples[neg_val_end:]

        random.shuffle(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """从内存缓存读取 + 数据增强"""
        cached = self.cache.get(idx)
        if cached is None:
            return self._create_empty_sample()

        image = torch.from_numpy(cached['data']).unsqueeze(0)
        label = torch.tensor(cached['label'], dtype=torch.long)
        bbox = torch.from_numpy(cached['bbox'])

        # ✅ 训练时随机翻转
        if self.mode == 'train':
            if random.random() > 0.5:
                image = image.flip(-1)  # 左右翻转
            if random.random() > 0.5:
                image = image.flip(-2)  # 上下翻转
            if random.random() > 0.5:
                image = image.flip(-3)  # 前后翻转

        return {
            'image': image,
            'label': label,
            'bbox': bbox
        }

    def _create_empty_sample(self) -> Dict[str, torch.Tensor]:
        return {
            'image': torch.zeros(1, 64, 64, 64),
            'label': torch.tensor(0, dtype=torch.long),
            'bbox': torch.zeros(6)
        }

class CTScanDataset(BaseCTDataset):
    """原始CT扫描数据集（用于推理）"""

    def __init__(self, ct_paths: List[Path], annotations_df: pd.DataFrame = None,
                 config: Dict = None, transform: Optional[tio.Transform] = None):
        super().__init__(config, transform)

        self.ct_paths = ct_paths
        self.annotations_df = annotations_df or load_annotations()

        self._load_samples()

        print(f"CT扫描数据集: {len(self.samples)}个CT扫描")

    def _load_samples(self):
        """加载CT扫描信息"""
        for ct_path in self.ct_paths:
            seriesuid = ct_path.stem
            ct_annotations = None
            if self.annotations_df is not None:
                ct_annotations = self.annotations_df[self.annotations_df['seriesuid'] == seriesuid]

            self.samples.append({
                'ct_path': ct_path,
                'seriesuid': seriesuid,
                'annotations': ct_annotations
            })

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个CT扫描"""
        sample_info = self.samples[idx]

        try:
            ct_array, spacing, origin, direction = self.preprocessor.load_ct_scan(sample_info['ct_path'])

            if ct_array is None:
                return self._create_empty_ct_sample(sample_info['seriesuid'])

            # 修复：resample_ct_monai 现在只需要一个参数
            resampled_array = self.preprocessor.resample_ct_monai(ct_array)
            normalized_array = self.preprocessor.normalize_ct_monai(resampled_array)
            image = torch.from_numpy(normalized_array).float().unsqueeze(0)

            annotations = []
            if sample_info['annotations'] is not None:
                for _, row in sample_info['annotations'].iterrows():
                    annotations.append({
                        'coordX': float(row['coordX']),
                        'coordY': float(row['coordY']),
                        'coordZ': float(row['coordZ']),
                        'diameter_mm': float(row['diameter_mm'])
                    })

            return {
                'image': image,
                'seriesuid': sample_info['seriesuid'],
                'original_shape': ct_array.shape,
                'processed_shape': normalized_array.shape,
                'spacing': spacing,
                'origin': origin,
                'direction': direction,
                'annotations': annotations,
                'ct_path': str(sample_info['ct_path'])
            }

        except Exception as e:
            print(f"加载CT扫描失败 {sample_info['seriesuid']}: {e}")
            return self._create_empty_ct_sample(sample_info['seriesuid'])

    def _create_empty_ct_sample(self, seriesuid: str) -> Dict[str, Any]:
        """创建空CT样本"""
        return {
            'image': torch.zeros(1, 128, 128, 128),
            'seriesuid': seriesuid,
            'original_shape': (128, 128, 128),
            'processed_shape': (128, 128, 128),
            'spacing': np.array([1.0, 1.0, 1.0]),
            'origin': np.array([0.0, 0.0, 0.0]),
            'direction': np.eye(3).flatten(),
            'annotations': [],
            'ct_path': 'error'
        }

class SlidingWindowDataset(BaseCTDataset):
    """滑动窗口数据集"""

    def __init__(self, ct_paths: List[Path], annotations_df: pd.DataFrame = None,
                 config: Dict = None, transform: Optional[tio.Transform] = None,
                 stride: int = 64):
        super().__init__(config, transform)

        self.ct_paths = ct_paths
        self.annotations_df = annotations_df or load_annotations()
        self.stride = stride
        self.crop_size = config.get("rpn_crop_size", [96, 96, 96])

        self._generate_windows()

        print(f"滑动窗口数据集: {len(self.windows)}个窗口")

    def _generate_windows(self):
        """生成滑动窗口"""
        self.windows = []

        for ct_idx, ct_path in enumerate(self.ct_paths[:5]):
            try:
                ct_array, spacing, origin, direction = self.preprocessor.load_ct_scan(ct_path)

                if ct_array is None:
                    continue

                resampled_array = self.preprocessor.resample_ct_monai(ct_array)
                normalized_array = self.preprocessor.normalize_ct_monai(resampled_array)

                seriesuid = ct_path.stem
                ct_annotations = self.annotations_df[self.annotations_df['seriesuid'] == seriesuid]

                ct_shape = normalized_array.shape

                for z in range(0, ct_shape[0] - self.crop_size[0] + 1, self.stride):
                    for y in range(0, ct_shape[1] - self.crop_size[1] + 1, self.stride):
                        for x in range(0, ct_shape[2] - self.crop_size[2] + 1, self.stride):
                            self.windows.append({
                                'ct_idx': ct_idx,
                                'ct_path': ct_path,
                                'seriesuid': seriesuid,
                                'window_pos': (z, y, x),
                                'window_size': self.crop_size,
                                'ct_shape': ct_shape,
                                'annotations': ct_annotations
                            })

            except Exception as e:
                print(f"生成窗口失败 {ct_path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个窗口"""
        window_info = self.windows[idx]

        try:
            ct_array, spacing, origin, direction = self.preprocessor.load_ct_scan(window_info['ct_path'])

            if ct_array is None:
                return self._create_empty_window()

            # 修复：resample_ct_monai 现在只需要一个参数
            resampled_array = self.preprocessor.resample_ct_monai(ct_array)
            normalized_array = self.preprocessor.normalize_ct_monai(resampled_array)
            
            z, y, x = window_info['window_pos']
            crop_size = window_info['window_size']
            
            window_data = normalized_array[
                z:z+crop_size[0],
                y:y+crop_size[1],
                x:x+crop_size[2]
            ]
            
            image = torch.from_numpy(window_data).float().unsqueeze(0)
            label, bbox = self._compute_window_label(window_info, normalized_array.shape)
            
            if self.torchio_transform is not None:
                augmented = self.apply_augmentation(image, label.unsqueeze(0), bbox.unsqueeze(0))
                image = augmented['image']
                label = augmented['label'].squeeze(0)
                bbox = augmented['bbox'].squeeze(0)
            
            return {
                'image': image,
                'label': label,
                'bbox': bbox,
                'has_bbox': torch.tensor([label.item() == 1]),
                'seriesuid': window_info['seriesuid'],
                'window_pos': torch.tensor(window_info['window_pos']),
                'window_size': torch.tensor(window_info['window_size'])
            }
            
        except Exception as e:
            print(f"加载窗口失败 {window_info['seriesuid']}: {e}")
            return self._create_empty_window()
    
    def _compute_window_label(self, window_info: Dict, ct_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算窗口标签"""
        z, y, x = window_info['window_pos']
        crop_size = window_info['window_size']
        
        window_center = np.array([
            z + crop_size[0] // 2,
            y + crop_size[1] // 2,
            x + crop_size[2] // 2
        ])
        
        has_nodule = False
        best_bbox = np.zeros(6)
        
        if window_info['annotations'] is not None:
            for _, row in window_info['annotations'].iterrows():
                world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
                voxel_coord = world_coord.astype(int)
                
                if (z <= voxel_coord[0] < z + crop_size[0] and
                    y <= voxel_coord[1] < y + crop_size[1] and
                    x <= voxel_coord[2] < x + crop_size[2]):
                    
                    has_nodule = True
                    rel_center = voxel_coord - np.array([z, y, x])
                    diameter = row['diameter_mm']
                    
                    best_bbox = np.array([
                        rel_center[0], rel_center[1], rel_center[2],
                        diameter, diameter, diameter
                    ])
                    
                    break
        
        if has_nodule:
            label = torch.tensor([1], dtype=torch.long)
            bbox = torch.from_numpy(best_bbox).float()
        else:
            label = torch.tensor([0], dtype=torch.long)
            bbox = torch.zeros(6).float()
        
        return label, bbox
    
    def _create_empty_window(self) -> Dict[str, torch.Tensor]:
        """创建空窗口"""
        return {
            'image': torch.zeros(1, *self.crop_size),
            'label': torch.tensor([0]),
            'bbox': torch.zeros(6),
            'has_bbox': torch.tensor([False]),
            'seriesuid': 'error',
            'window_pos': torch.zeros(3),
            'window_size': torch.tensor(self.crop_size)
        }

def create_rpn_dataloader(data_dir: Path = None, mode: str = 'train',
                          batch_size: int = 4, num_workers: int = 4,
                          config: Dict = None) -> torch.utils.data.DataLoader:
    """创建RPN数据加载器"""
    dataset = RPNDataset(data_dir=data_dir, mode=mode, config=config)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    return dataloader

def create_classifier_dataloader(data_dir: Path = None, mode: str = 'train',
                                 batch_size: int = 8, num_workers: int = 4,
                                 config: Dict = None) -> torch.utils.data.DataLoader:
    """创建分类器数据加载器"""
    dataset = ClassifierDataset(data_dir=data_dir, mode=mode, config=config)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    print("测试数据集模块...")
    
    print("\n1. 测试RPN数据集...")
    try:
        rpn_dataset = RPNDataset(mode='train')
        print(f"RPN数据集大小: {len(rpn_dataset)}")
        
        if len(rpn_dataset) > 0:
            sample = rpn_dataset[0]
            print(f"样本键: {list(sample.keys())}")
            print(f"图像形状: {sample['image'].shape}")
            print(f"标签: {sample['label']}")
    except Exception as e:
        print(f"RPN数据集测试失败: {e}")
    
    print("\n2. 测试分类器数据集...")
    try:
        classifier_dataset = ClassifierDataset(mode='train')
        print(f"分类器数据集大小: {len(classifier_dataset)}")
        
        if len(classifier_dataset) > 0:
            sample = classifier_dataset[0]
            print(f"样本键: {list(sample.keys())}")
            print(f"图像形状: {sample['image'].shape}")
            print(f"标签: {sample['label']}")
    except Exception as e:
        print(f"分类器数据集测试失败: {e}")
    
    print("数据集模块测试完成!")