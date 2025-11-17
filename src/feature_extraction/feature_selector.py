"""
特征选择模块

使用随机森林进行数据驱动的特征重要性评估和选择
"""
import logging
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


class FeatureSelector:
    """静态特征选择器，用于优化DCT特征"""

    @staticmethod
    def select_important_features(
        dct_features: np.ndarray,
        target_dim: int = 500,
        method: str = 'rf',
        random_state: int = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用指定方法选择重要的DCT系数

        Args:
            dct_features: DCT特征，支持2D格式 (N, feature_dim) 或4D格式 (N, channels, height, width)
            target_dim: 目标特征维度
            method: 选择方法 ('rf', 'pca', 'variance', 'univariate')
            random_state: 随机种子

        Returns:
            选择后的特征和重要性indices
        """
        if len(dct_features.shape) == 4:
            N, channels, height, width = dct_features.shape
            dct_features_2d = dct_features.reshape(
                N, channels * height * width)
            logger.info(
                f"输入4D数据 {dct_features.shape} 转换为2D: {dct_features_2d.shape}")
        elif len(dct_features.shape) == 2:
            dct_features_2d = dct_features
            logger.info(f"输入2D数据: {dct_features_2d.shape}")
        else:
            raise ValueError(
                f"不支持的输入格式，期望2D (N, features) 或4D (N, channels, height, width)，得到: {dct_features.shape}")

        logger.info(f"开始特征选择，原始特征维度: {dct_features_2d.shape[1]}")
        logger.info(f"目标特征维度: {target_dim}")

        if method == 'rf':
            selected_features, selector_info = FeatureSelector._random_forest_selection(
                dct_features_2d, target_dim, random_state)
        elif method == 'pca':
            selected_features, selector_info = FeatureSelector._pca_selection(
                dct_features_2d, target_dim, random_state)
        elif method == 'variance':
            selected_features, selector_info = FeatureSelector._variance_selection(
                dct_features_2d, target_dim)
        elif method == 'univariate':
            selected_features, selector_info = FeatureSelector._univariate_selection(
                dct_features_2d, target_dim, random_state)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        logger.info(f"特征选择完成，最终特征维度: {selected_features.shape[1]}")

        return selected_features, selector_info

    @staticmethod
    def _random_forest_selection(dct_features: np.ndarray, target_dim: int, random_state: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用随机森林进行特征选择

        Args:
            dct_features: DCT特征 (N, feature_dim)
            target_dim: 目标特征维度
            random_state: 随机种子

        Returns:
            选择后的特征和索引
        """
        logger.info("使用随机森林方法进行特征选择")

        # 生成伪标签进行无监督特征选择
        n_clusters = min(50, len(dct_features) // 100, len(dct_features) // 2)
        if n_clusters < 2:
            n_clusters = 2

        logger.info(f"生成 {n_clusters} 个伪标签进行特征评估")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(1000, len(dct_features))
        )
        pseudo_labels = kmeans.fit_predict(dct_features)

        # 随机森林特征重要性评估
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            max_depth=10  # 限制深度避免过拟合
        )

        logger.info("训练随机森林评估特征重要性")
        rf.fit(dct_features, pseudo_labels)

        # 选择最重要的特征
        importances = rf.feature_importances_
        important_indices = np.argsort(importances)[-target_dim:]

        selected_features = dct_features[:, important_indices]
        rf_info = {
            'method': 'rf',
            'important_indices': important_indices,
            'random_state': random_state
        }
        return selected_features, rf_info

    @staticmethod
    def _pca_selection(dct_features: np.ndarray, target_dim: int, random_state: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用PCA进行特征降维

        Args:
            dct_features: DCT特征 (N, feature_dim)
            target_dim: 目标特征维度
            random_state: 随机种子

        Returns:
            降维后的特征和 PCA 变换信息（用于后续新数据变换）
        """
        logger.info("使用PCA方法进行特征降维")

        # 标准化特征
        mean_vals = np.mean(dct_features, axis=0)
        std_vals = np.std(dct_features, axis=0)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)

        normalized_features = (dct_features - mean_vals) / std_vals

        # PCA降维
        pca = PCA(n_components=target_dim, random_state=random_state)
        transformed_features = pca.fit_transform(normalized_features)

        # 保存 PCA 变换信息
        pca_info = {
            'method': 'pca',
            'mean_vals': mean_vals,
            'std_vals': std_vals,
            'pca_components': pca.components_,
            'n_components': target_dim,
            'random_state': random_state
        }

        return transformed_features, pca_info

    @staticmethod
    def _variance_selection(dct_features: np.ndarray, target_dim: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        基于方差进行特征选择

        Args:
            dct_features: DCT特征 (N, feature_dim)
            target_dim: 目标特征维度

        Returns:
            选择后的特征和索引
        """
        logger.info("使用方差方法进行特征选择")

        # 计算每个特征的方差
        variances = np.var(dct_features, axis=0)

        # 选择方差最大的特征
        important_indices = np.argsort(variances)[-target_dim:]
        selected_features = dct_features[:, important_indices]
        variance_info = {
            'method': 'variance',
            'important_indices': important_indices
        }
        return selected_features, variance_info

    @staticmethod
    def _univariate_selection(dct_features: np.ndarray, target_dim: int, random_state: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用单变量统计测试进行特征选择

        Args:
            dct_features: DCT特征 (N, feature_dim)
            target_dim: 目标特征维度
            random_state: 随机种子

        Returns:
            选择后的特征和索引
        """
        logger.info("使用单变量统计测试进行特征选择")

        # 生成伪标签
        n_clusters = min(20, len(dct_features) // 50, len(dct_features) // 2)
        if n_clusters < 2:
            n_clusters = 2

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state)
        pseudo_labels = kmeans.fit_predict(dct_features)

        # 单变量特征选择
        selector = SelectKBest(f_classif, k=target_dim)
        selected_features = selector.fit_transform(dct_features, pseudo_labels)

        # 获取选择的特征索引
        important_indices = selector.get_support(indices=True)
        univariate_info = {
            'method': 'univariate',
            'important_indices': important_indices
        }
        return selected_features, univariate_info

    @staticmethod
    def transform_features(features: np.ndarray, selected_info: Dict[str, Any]) -> np.ndarray:
        """
        对特征应用特征索引选择

        Args:
            features: 特征 (N, original_dim)
            selected_info: 特征选择信息

        Returns:
            转换后的特征 (N, selected_dim)
        """
        if selected_info is None:
            logger.info("未使用特征选择器")
            raise ValueError("selected_info 不能为空")

        method = selected_info.get('method', None)
        if method == 'rf':
            return features[:, selected_info['important_indices']]
        elif method == 'pca':
            mean_vals = selected_info['mean_vals']
            std_vals = selected_info['std_vals']
            components = selected_info['pca_components']
            # 标准化
            normalized = (features - mean_vals) / std_vals
            # 投影到主成分空间
            transformed = np.dot(normalized, components.T)
            return transformed
        elif method == 'variance':
            return features[:, selected_info['important_indices']]
        elif method == 'univariate':
            return features[:, selected_info['important_indices']]
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

    @staticmethod
    def transform_features_pca(features: np.ndarray, pca_info: Dict[str, Any]) -> np.ndarray:
        """
        使用PCA信息转换特征

        Args:
            features: 特征 (N, original_dim)
            pca_info: PCA 变换信息
        Returns:
            转换后的特征 (N, selected_dim)
        """
        mean_vals = pca_info['mean_vals']
        std_vals = pca_info['std_vals']
        components = pca_info['pca_components']
        # 标准化
        normalized = (features - mean_vals) / std_vals
        # 投影到主成分空间
        transformed = np.dot(normalized, components.T)
        return transformed
