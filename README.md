# 游戏资产图片聚类与相似匹配系统

本项目用于大规模游戏图片聚类与相似图片查找，适合资源查重、分包、冗余清理等场景。支持 RGBA 四通道，自适应权重平衡透明通道。

## 环境配置

1. 推荐使用 Python 3.8 及以上版本
2. 建议使用虚拟环境隔离依赖

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 项目配置说明

### 快速开始

编辑 `config/settings.yaml`，根据你的项目调整以下关键参数：

1. **数据源配置** - 设置游戏资产图片目录
   ```yaml
   data_sources:
     asset_directories:
       - "C:/YourProjectPath/A"
       - "C:/YourProjectPath/B"  # 支持多目录
   ```

2. **特征提取参数** - 平衡精度与速度，通常不需要调整
   ```yaml
   feature_extraction:
     image_size: 256          # 预处理统一尺寸，越大特征保留越多细节
     dct_size: 32             # DCT低频分量尺寸，影响特征维度
     feature_dim: 500         # 最终特征维度，降维后特征数
   ```

3. **聚类参数** - 根据图片数量调整
   ```yaml
   clustering:
     n_clusters: 4000         # 聚类数 ≈ 图片总数 / 15
     batch_size: 1000         # 批处理大小，根据内存调整
   ```

4. **匹配参数** - 调整搜索策略
   ```yaml
   matching:
     default_top_k: 10                   # 单张查询返回相似图片数
     max_search_clusters: 1              # 搜索簇数，1=仅簇内搜索，>1=跨簇扩展
     similarity_threshold: 0.9            # 报告相似度阈值
   ```

5. **性能配置**
   ```yaml
   performance:
     cache_dir: "./cache"    # 特征与模型缓存目录
   ```

## 使用方法

### 1. 首次初始化：提取特征并聚类

```bash
python app.py update --config config/settings.yaml
```

此步将：
- 递归扫描资产目录中的所有图片
- 提取 DCT 特征（支持 RGBA，自适应 Alpha 权重）
- 进行特征降维（随机森林/PCA/方差选择）
- 构建 KMeans 聚类模型
- 建立 BallTree 空间索引
- 缓存所有中间结果，后续查询快速响应

### 2. 查找单张图片的相似资源

```bash
python app.py query --image "C:/path/to/image.png" --top-k 5
```

### 3. 生成全量相似图片报告

```bash
python app.py report
```

输出 HTML 报告到 `reports/` 目录，包含：
- 所有相似图片组
- 图片预览缩略图
- 相似度分数
- 文件路径及大小

### 4. 开发/测试环境

小规模测试推荐调整参数：
```yaml
feature_extraction:
  image_size: 128    # 降低分辨率加速
  feature_dim: 200   # 减少特征维度
clustering:
  n_clusters: 100    # 小数据集使用少量聚类
```

## 目录结构说明

```
image_matching_dct/
├── src/
│   ├── feature_extraction/    DCT 特征提取、通道权重、特征选择
│   ├── clustering/            KMeans 聚类、BallTree 索引
│   ├── matching/              相似搜索、结果排序
│   └── apis/                  CLI 接口、报告生成
├── config/                    YAML 配置文件
├── cache/                     特征缓存、模型缓存
├── reports/                   HTML 查重报告
└── requirements.txt           依赖包列表
```

## 性能优化建议

1. **大规模项目（10000+ 图片）**
   - 增加 `batch_size` 至 2000-5000 以提升吞吐量
   - 减少 `feature_dim` 至 300-400 以加速聚类
   - 调整 `n_clusters` 为 √(图片数) 左右

2. **精度优先场景**
   - 增加 `image_size` 至 512（保留更多细节）
   - 增加 `feature_dim` 至 800-1000
   - 降低 `similarity_threshold` 以提升召回率

3. **首次聚类速度慢**
   - 此为正常现象，特征提取与聚类需要时间
   - 后续查询将从缓存直接读取

## 常见问题

- **Q: 如何处理新增图片？**
  A: 重新运行 `cluster` 命令，系统自动增量更新聚类和索引。

- **Q: 内存不足怎么办？**
  A: 减小 `batch_size` 或 `image_size`，或分批处理。

- **Q: 查询结果不满意？**
  A: 调整 `similarity_threshold` 或增加 `max_search_clusters` 扩大搜索范围。

如遇其他问题，检查 `cache/log/` 日志文件获取详细错误信息。

