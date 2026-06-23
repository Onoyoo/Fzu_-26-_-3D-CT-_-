该项目是FZU AI实践课的小组作业的主要代码部分。

实现了一个 RPN + 分类器 的二阶段肺部病灶检测模型。

---

luna16_project/

│

├── config.py           # 全局配置文件（路径、超参数、模型配置）

│

├── raw/              # 原始 CT 数据（.mhd/.raw）

│  ├── subset0/          # 子集0（约89套CT）

│  ├── subset1/          # 子集1（约89套CT）

│  └── ......           # subset0-9 共10个文件夹

│

├── csv/              # 标注和CSV文件

│  ├── annotations.csv      # 1186个标准结点

│  └──candidates.csv       # 55万个候选点（分类器训练用）

│

├── codes/             # 源代码

│  ├── datasets/         # 数据加载模块

│  │  ├── luna16_dataset.py   # PyTorch Dataset类

│  │  └── preprocess.py     # 预处理（重采样/截断/归一化）

│  │

│  ├── models/          # 模型定义模块

│  │  ├── rpn.py         # 3D RPN 网络

│  │  ├── classifier.py     # 3D CNN 分类器

│  │  └── attention.py      # 注意力机制模块

│  │

│  ├── losses/          # 损失函数模块

│  │  └── focal_loss.py     # Focal Loss 实现

│  │

│  ├── train/           # 训练脚本

│  │  ├── train_rpn.py      # RPN 训练主脚本

│  │  └──train_classifier.py  # 分类器训练主脚本

│  │

│  ├── inference/         # 推理模块

│  │  ├── inference.py      # 推理主脚本

└  └  └── postprocess.py     # 后处理（NMS）

---