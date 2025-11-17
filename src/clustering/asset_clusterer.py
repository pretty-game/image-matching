"""
游戏资产聚类模块

使用MiniBatchKMeans和BallTree索引实现大规模图像聚类
"""
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import BallTree, NearestNeighbors

logger = logging.getLogger(__name__)


class GameAssetClusterer:
    """静态游戏资产聚类器，专门处理大规模游戏图片聚类"""

    @staticmethod
    def cluster_assets(
        features: np.ndarray,
        n_clusters: int = 300,
        batch_size: int = 1000,
        random_state: int = 42
    ) -> Tuple[np.ndarray, Any]:
        """
        聚类游戏资产图片

        Args:
            features: 特征数组 (N, feature_dim)
            image_paths: 图片路径列表
            n_clusters: 聚类数量
            batch_size: 批处理大小
            random_state: 随机种子

        Returns:
            聚类标签、KMeans模型
        """
        n_samples = len(features)
        effective_n_clusters = n_clusters
        max_samples = n_samples // 2
        if max_samples < n_clusters:
            effective_n_clusters = max(1, min(n_samples, max_samples))
            logger.warning(f"样本数量/2 ({max_samples}) < 配置的聚类数({n_clusters})，自动调整为 {effective_n_clusters} 个聚类")

        logger.info(f"开始聚类 {n_samples} 张图片到 {effective_n_clusters} 个簇")

        # 根据数据量选择聚类算法
        if len(features) > 10000:
            logger.info("使用MiniBatchKMeans处理大规模数据")
            kmeans = MiniBatchKMeans(
                n_clusters=effective_n_clusters,
                batch_size=batch_size,
                random_state=random_state,
                n_init=3,
                max_iter=100,
                reassignment_ratio=0.01
            )
        else:
            logger.info("使用标准KMeans处理小规模数据")
            kmeans = KMeans(
                n_clusters=effective_n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )

        # 执行聚类
        cluster_labels = kmeans.fit_predict(features)
        logger.info("聚类完成")

        return cluster_labels, kmeans

    @staticmethod
    def build_cluster_index(labels: np.ndarray, paths: List[str], features: np.ndarray) -> Dict[int, Dict]:
        """
        为每个簇构建搜索索引

        Args:
            labels: 聚类标签 (N,)
            paths: 图片路径列表 (N,)
            features: 特征数组 (N, feature_dim)

        Returns:
            聚类索引字典
        """
        logger.info("构建BallTree索引...")

        cluster_index = {}
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            cluster_mask = (labels == cluster_id)
            cluster_features = features[cluster_mask]
            cluster_paths = [paths[i] for i in np.where(cluster_mask)[0]]

            if len(cluster_features) > 0:
                # 为每个簇构建BallTree索引，确保数据类型为float64
                cluster_features_float64 = cluster_features.astype(np.float64)
                tree = BallTree(cluster_features_float64, metric='euclidean')

                cluster_index[cluster_id] = {
                    'tree': tree,
                    'features': cluster_features_float64,  # 保存float64版本
                    'paths': cluster_paths,
                    'center': np.mean(cluster_features_float64, axis=0),
                    'size': len(cluster_features),
                    'std': np.std(cluster_features_float64, axis=0)
                }

        logger.info(f"为 {len(unique_labels)} 个簇构建了索引")
        return cluster_index

    @staticmethod
    def supports_incremental_update(kmeans_model) -> bool:
        """
        检查聚类模型是否支持增量更新

        Args:
            kmeans_model: KMeans或MiniBatchKMeans模型

        Returns:
            是否支持partial_fit增量更新
        """
        return hasattr(kmeans_model, "partial_fit")

    # 增量更新cluster
    @staticmethod
    def update_clusterer_incremental(
        kmeans_model,
        cluster_labels: np.ndarray,
        image_paths: List[str],
        selected_features: np.ndarray,
        deleted_image_paths: Optional[List[str]] = None,
        added_image_paths: Optional[List[str]] = None,
        added_features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Any, np.ndarray, List[str]]:
        """
        增量更新聚类模型，支持新增、删除（修改操作由调用方拆分为删除+新增）

        Args:
            kmeans_model: 现有KMeans或MiniBatchKMeans模型
            cluster_labels: 现有聚类标签 (N,)
            image_paths: 所有样本路径 (N,)
            features: 所有样本的特征 (N, feature_dim)
            deleted_image_paths: 删除的样本路径列表
            added_image_paths: 新增的样本路径列表
            added_features: 新增样本的特征数组 (M, feature_dim)

        Returns:
            更新后的聚类标签、KMeans模型、特征数组、路径列表
        """
        logger.info("开始增量更新聚类模型...")

        # 初始化参数
        deleted_image_paths = deleted_image_paths or []
        added_image_paths = added_image_paths or []
        added_features = added_features if added_features is not None else np.empty((0, selected_features.shape[1]))

        logger.info(f"新增样本数: {len(added_image_paths)}, 删除样本数: {len(deleted_image_paths)}")

        # 1. 删除样本
        if len(deleted_image_paths) > 0:
            remove_paths = set(deleted_image_paths)
            keep_indices = [i for i, p in enumerate(image_paths) if p not in remove_paths]
            cluster_labels = cluster_labels[keep_indices]
            selected_features = selected_features[keep_indices]
            image_paths = [image_paths[i] for i in keep_indices]
            logger.info(f"已移除 {len(deleted_image_paths)} 个样本")

        # 2. 新增样本
        if len(added_features) > 0:
            logger.info(f"为 {len(added_features)} 个新样本分配聚类标签")
            new_labels = kmeans_model.predict(added_features)
            cluster_labels = np.concatenate([cluster_labels, new_labels], axis=0)
            selected_features = np.vstack([selected_features, added_features])
            image_paths = image_paths + added_image_paths

            # MiniBatchKMeans支持partial_fit进行微调
            if hasattr(kmeans_model, "partial_fit"):
                logger.info("使用MiniBatchKMeans.partial_fit进行微调")
                kmeans_model.partial_fit(added_features)
            else:
                logger.info("KMeans不支持partial_fit，仅分配标签，不更新模型参数")

        logger.info("增量更新完成")
        return cluster_labels, kmeans_model, selected_features, image_paths


    @staticmethod
    def compute_cluster_statistics(
        features: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        kmeans_model: Any
    ) -> Dict[str, Any]:
        """
        计算聚类统计信息

        Args:
            features: 特征数组
            labels: 聚类标签
            n_clusters: 聚类数量
            kmeans_model: KMeans模型

        Returns:
            聚类统计信息
        """
        stats = {}

        # 基本统计
        stats['n_clusters'] = n_clusters
        stats['n_samples'] = len(features)
        stats['feature_dim'] = features.shape[1]
        stats['timestamp'] = datetime.now().isoformat()

        # 聚类质量评估
        try:
            if len(np.unique(labels)) > 1:
                stats['silhouette_score'] = silhouette_score(features, labels)
                stats['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
        except Exception as e:
            logger.warning(f"计算聚类质量指标失败: {e}")

        # 簇大小分布
        unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
        stats['cluster_sizes'] = dict(zip(unique_labels.tolist(), cluster_sizes.tolist()))
        stats['avg_cluster_size'] = np.mean(cluster_sizes)
        stats['std_cluster_size'] = np.std(cluster_sizes)
        stats['min_cluster_size'] = np.min(cluster_sizes)
        stats['max_cluster_size'] = np.max(cluster_sizes)

        # 簇间距离统计
        if kmeans_model is not None and hasattr(kmeans_model, 'cluster_centers_'):
            centers = kmeans_model.cluster_centers_
            # 计算簇中心间的距离（仅在有多个簇时）
            if len(centers) > 1:
                from scipy.spatial.distance import pdist
                center_distances = pdist(centers, metric='euclidean')
                stats['avg_inter_cluster_distance'] = np.mean(center_distances)
                stats['min_inter_cluster_distance'] = np.min(center_distances)
                stats['max_inter_cluster_distance'] = np.max(center_distances)
            else:
                # 只有一个簇时，无法计算簇间距离
                stats['avg_inter_cluster_distance'] = 0.0
                stats['min_inter_cluster_distance'] = 0.0
                stats['max_inter_cluster_distance'] = 0.0

        return stats

    @staticmethod
    def get_cluster_neighbors(kmeans_model: Any, cluster_index: Dict[int, Dict], cluster_id: int, n_neighbors: int = 5) -> List[int]:
        """
        获取指定簇的邻近簇

        Args:
            kmeans_model: KMeans模型
            cluster_index: 聚类索引
            cluster_id: 目标簇ID
            n_neighbors: 邻居数量

        Returns:
            邻近簇ID列表
        """
        if (kmeans_model is None or
            not hasattr(kmeans_model, 'cluster_centers_') or
            cluster_index is None or
                cluster_id not in cluster_index):
            return []

        centers = kmeans_model.cluster_centers_
        target_center = centers[cluster_id].reshape(1, -1)

        # 使用NearestNeighbors找到最近的簇中心
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(centers)), metric='euclidean')
        nn.fit(centers)
        distances, indices = nn.kneighbors(target_center)

        # 排除自己，返回邻居簇ID
        neighbor_ids = [int(idx) for idx in indices[0][1:]]

        return neighbor_ids

    @staticmethod
    def find_nearest_clusters(kmeans_model: Any, query_features: np.ndarray, n_clusters: int = 3) -> List[Tuple[int, float]]:
        """
        找到最接近查询特征的簇

        Args:
            kmeans_model: KMeans模型
            query_features: 查询特征 (feature_dim,)
            n_clusters: 返回的簇数量

        Returns:
            (簇ID, 距离)的列表，按距离排序
        """
        if kmeans_model is None or not hasattr(kmeans_model, 'cluster_centers_'):
            return []

        centers = kmeans_model.cluster_centers_.astype(np.float64)
        query_features_float64 = query_features.astype(np.float64)

        # 计算到所有簇中心的距离
        distances = np.linalg.norm(centers - query_features_float64, axis=1)

        # 获取最近的簇
        nearest_indices = np.argsort(distances)[:n_clusters]

        result = [(int(idx), float(distances[idx])) for idx in nearest_indices]

        return result

    @staticmethod
    def get_cluster_summary(cluster_index: Dict[int, Dict], kmeans_model: Any, cluster_id: int) -> Dict[str, Any]:
        """
        获取指定簇的详细信息

        Args:
            cluster_index: 聚类索引
            kmeans_model: KMeans模型
            cluster_id: 簇ID

        Returns:
            簇的详细信息
        """
        if cluster_index is None or cluster_id not in cluster_index:
            return {}

        cluster_data = cluster_index[cluster_id]

        summary = {
            'cluster_id': cluster_id,
            'size': cluster_data['size'],
            'sample_paths': cluster_data['paths'][:5],  # 显示前5个样本
            'center_norm': np.linalg.norm(cluster_data['center']),
            'avg_std': np.mean(cluster_data['std']),
            'feature_dim': len(cluster_data['center'])
        }

        # 获取邻近簇
        neighbors = GameAssetClusterer.get_cluster_neighbors(kmeans_model, cluster_index, cluster_id, n_neighbors=3)
        summary['neighbor_clusters'] = neighbors

        return summary
