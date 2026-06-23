"""配置文件 - LUNA16项目"""

from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.absolute()

RAW_DATA_DIR = PROJECT_ROOT / "raw"
CSV_DIR = PROJECT_ROOT / "csv"
PROCESSED_DIR = PROJECT_ROOT / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

MODELS_DIR = PROJECT_ROOT / "models"
RPN_MODEL_PATH = MODELS_DIR / "rpn" / "rpn_best.pth"

# ==================== 数据配置 ====================
DATA_CONFIG = {
    "voxel_spacing": [0.76, 0.76, 2.5],
    "rpn_crop_size": [96, 96, 96],
    "classifier_crop_size": [64, 64, 64],
    "ct_clip_min": -1000,
    "ct_clip_max": 400,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "augmentation": {
        "rotation_range": 15,
        "scale_range": 0.1,
        "flip_prob": 0.5,
        "elastic_deformation": True,
        "random_bias_field": True,
        "random_noise": True,
        "intensity_transform": True
    }
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    "use_monai": True,
    "rpn": {
        "backbone": "unet3d",
        "anchor_sizes": [16, 32, 64],
        "anchor_ratios": [0.5, 1.0, 2.0],
        "feature_channels": 128,
        "dropout_rate": 0.2,
        "pretrained": True,
        "freeze_backbone": False
    },
    "classifier": {
        "backbone": "simple",
        "input_channels": 1,
        "num_classes": 2,
        "attention": True,
        "attention_type": "se",
        "dropout_rate": 0.3,
        "pretrained": True,
        "freeze_backbone": True
    },
    "monai": {
        "unet3d": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 256,
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2),
            "num_res_units": 2
        },
        "densenet3d": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 576,
                "init_features": 64,
                "growth_rate": 32,
                "block_config": (6, 12, 24, 16),
                "bn_size": 4,
                "dropout_prob": 0.3
            }
        }
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    "batch_size": 4, 
    "num_workers": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "save_freq": 10,
    "optimizer": "adamw",
    "scheduler": "cosineannealing",
    "loss_functions": {
        "rpn_cls": "focal_loss",
        "rpn_reg": "smooth_l1",
        "cls": "dice_focal"
    },
    "loss_weights": {
        "rpn_cls": 1.0,
        "rpn_reg": 1.0,
        "cls": 1.0
    },
    "monai": {
        "amp": False,
        "gradient_clip": 5.0,
        "early_stopping": True,
        "patience": 10,
        "metric": "froc",
        "metric_save_best": True
    }
}

# ==================== 推理配置 ====================
INFERENCE_CONFIG = {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.3,
    "max_detections": 100,
    "batch_size": 1
}