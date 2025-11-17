# 游戏资产图片聚类与匹配系统 - AI 开发指南

## 项目概览
这是一个为50000张游戏资产图片（RGBA格式）设计的聚类与相似匹配系统。项目当前处于**设计阶段** - README.md包含完整架构设计，但实际实现尚未开始。

## 核心架构理念
系统采用**分层模块化**设计，专门优化处理游戏资源特性：
- **RGBA四通道特征提取**：使用DCT变换，动态权重平衡Alpha透明通道
- **大规模数据处理**：MiniBatchKMeans + BallTree索引，支持50000+图片规模
- **两阶段搜索策略**：簇内精确搜索 + 跨簇扩展搜索提高召回率

## 实现时的关键决策

### 依赖管理策略
```python
# 当前dependencies (requirements.txt) 专注核心功能：
# - OpenCV: 图像处理主力（含stubs用于类型提示）
# - NumPy/SciPy: 数值计算基础
# - Pillow: RGBA图像加载与预处理
# 缺失：sklearn, PyYAML - 需要在实现聚类时添加
```

### 模块结构约定
实现时严格遵循README中的目录结构：
- `src/feature_extraction/`: DCT特征提取 + 自适应通道权重
- `src/clustering/`: MiniBatchKMeans + BallTree索引构建
- `src/matching/`: 相似图片搜索引擎
- `config/`: YAML配置文件（特征维度、聚类参数等）

### 游戏资产特殊处理
```python
# Alpha通道权重自适应算法 (关键差异化特性)
def balance_channels(self, channel_features, original_img):
    alpha_usage = np.mean(original_img[:, :, 3] > 10)
    weights = [0.2, 0.2, 0.2, 0.4] if alpha_usage > 0.3 else [0.25, 0.25, 0.25, 0.25]
    # UI元素通常高Alpha使用率，贴图则相对均匀
```

### 性能优化模式
- **批处理策略**：1000张图片/批次避免内存溢出
- **特征缓存机制**：使用pickle缓存提取的DCT特征
- **增量学习支持**：新图片加入无需重新聚类整个数据集

## VS Code配置说明
`.vscode/settings.json` 已配置：
- **Pylint仅报错模式**：`"--errors-only"` 避免OpenCV等C扩展的误报
- **类型检查降级**：针对图像处理库的未知类型设为`"information"`级别
- **120字符行限制**：适配长特征提取管道的可读性

## 开发工作流

### 1. 实现顺序建议
```bash
# 首先建立项目骨架
mkdir -p src/{feature_extraction,clustering,matching,utils} config tests examples

# 按依赖关系实现：预处理 → 特征提取 → 聚类 → 匹配
```

### 2. 测试策略
使用小规模样本（100-1000张图片）验证算法，再扩展到50000张：
```python
# examples/test_small_dataset.py - 先实现这个验证流程
asset_manager = GameAssetManager('config/test_settings.yaml')
```

### 3. 配置文件优先级
- `config/settings.yaml`: 生产环境（50000图片）
- `config/test_settings.yaml`: 开发测试（小规模数据）
- 运行时通过构造参数选择配置

## 实现注意事项

### 内存管理
```python
# 必须实现的批处理模式，防止50000张图片导致OOM
def process_large_dataset_in_batches(image_paths, batch_size=1000):
    for i in range(0, len(image_paths), batch_size):
        yield extract_features_batch(image_paths[i:i+batch_size])
```

### 错误处理模式
游戏资产可能包含损坏文件，需要robust的加载机制：
```python
# src/feature_extraction/preprocessor.py
def load_rgba_images(self, image_paths):
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGBA')
        except Exception as e:
            logger.warning(f"跳过损坏图片: {path}, 错误: {e}")
            continue
```

### 类型提示约定
利用已安装的stubs包，确保OpenCV调用有完整类型支持：
```python
import cv2
import numpy as np
from typing import List, Tuple, Optional

def calculate_dct(self, image_data: np.ndarray, dct_size: int = 64) -> np.ndarray:
```

### 虚拟环境
每次运行务必先启动venv虚拟环境

这个项目的核心价值在于将通用图像聚类算法**特化为游戏资产处理**，重点关注RGBA通道权重平衡和大规模数据的工程化处理。