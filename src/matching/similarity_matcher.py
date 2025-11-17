"""
相似图片匹配模块

实现基于聚类的两阶段搜索策略：簇内精确搜索 + 跨簇扩展搜索
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.feature_extraction.dct_extractor import DCTFeatureExtractor
from src.feature_extraction.feature_selector import FeatureSelector

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """相似性搜索结果"""
    path: str
    distance: float
    cluster_id: int
    rank: int
    confidence: float = 0.0
    similarity: float = 0.0  # 新增：计算得出的相似度分数


class SimilarityMatcher:
    """相似图片匹配引擎，基于聚类索引进行高效搜索"""

    @staticmethod
    def find_similar_images(
        cluster_model: Any,
        cluster_index: Optional[Dict[int, Dict]],
        selected_info: Optional[Dict[str, Any]],
        query_image_path: str,
        image_size: int,
        dct_size: int,
        top_k: int = 10,
        max_clusters: int = 3,
        similarity_threshold: float = 0.9
    ) -> List[SimilarityResult]:
        """
        查找相似图片（静态方法）

        Args:
            cluster_model: 聚类模型
            cluster_index: 聚类索引
            selected_info: 特征选择信息
            query_image_path: 查询图片路径
            image_size: 图片大小
            dct_size: DCT大小
            top_k: 返回的相似图片数量
            max_clusters: 最大搜索簇数量
            similarity_threshold: 相似性阈值

        Returns:
            相似图片结果列表
        """
        logger.info(f"查找与 {query_image_path} 相似的图片")

        if cluster_model is None:
            logger.error("聚类模型未加载，无法进行相似图片搜索")
            raise ValueError("cluster_model 不能为空")

        if selected_info is None:
            logger.info("未使用特征选择器")
            raise ValueError("selected_indices 不能为空")

        # 提取查询图片特征
        query_feature_data = DCTFeatureExtractor.extract_multi_channel_dct(
            query_image_path, image_size, dct_size)
        if query_feature_data is None:
            logger.error(f"无法提取查询图片特征: {query_image_path}")
            return []

        # 使用特征选择器处理特征（如果存在）
        dct_feature = query_feature_data['dct_features'].astype(np.float64)
        # 直接应用特征索引选择
        final_feature = FeatureSelector.transform_features(
            dct_feature.reshape(1, -1), selected_info).astype(np.float64).flatten()

        # 确定查询图片所属的簇，确保数据类型一致
        # 检查模型的数据类型，匹配模型期望的类型
        if hasattr(cluster_model, 'cluster_centers_'):
            model_dtype = cluster_model.cluster_centers_.dtype
            # 使用与模型相同的数据类型
            query_input = final_feature.reshape(1, -1).astype(model_dtype, copy=False)
        else:
            # 后备方案：使用float32
            query_input = final_feature.reshape(1, -1).astype(np.float32, copy=False)

        if not query_input.flags.c_contiguous:
            query_input = np.ascontiguousarray(query_input)

        query_cluster = cluster_model.predict(query_input)[0]
        logger.info(f"查询图片分配到簇: {query_cluster}")

        # 在所属簇内搜索
        similar_in_cluster = SimilarityMatcher.search_in_cluster(
            cluster_index, query_cluster, final_feature, top_k * 2)
        # 扩展搜索到相邻簇（提高召回率）
        all_similar = similar_in_cluster
        if max_clusters > 1:
            similar_expanded = SimilarityMatcher.expand_search(
                cluster_model, cluster_index, final_feature, top_k, max_clusters)
            all_similar = SimilarityMatcher.merge_results(
                similar_in_cluster, similar_expanded, top_k * 3)

        # 过滤和排序结果
        filtered_results = SimilarityMatcher._filter_and_rank_results(
            all_similar, query_image_path, top_k, similarity_threshold
        )

        logger.info(f"找到 {len(filtered_results)} 个相似图片")
        return filtered_results

    @staticmethod
    def search_in_cluster(cluster_index: Dict[int, Dict], cluster_id: int, query_features: np.ndarray, top_k: int) -> List[SimilarityResult]:
        """
        在指定簇内搜索相似图片

        Args:
            cluster_index: 聚类索引
            cluster_id: 目标簇ID
            query_features: 查询特征
            top_k: 返回数量

        Returns:
            相似图片结果列表
        """
        if cluster_id not in cluster_index:
            logger.warning(f"簇 {cluster_id} 不存在于索引中")
            return []

        cluster_data = cluster_index[cluster_id]

        # 使用BallTree进行快速近邻搜索，确保数据类型匹配
        k = min(top_k, len(cluster_data['features']))
        query_features_float64 = query_features.astype(
            np.float64).reshape(1, -1)
        distances, indices = cluster_data['tree'].query(
            query_features_float64, k=k)

        results = []
        for i, idx in enumerate(indices[0]):
            result = SimilarityResult(
                path=cluster_data['paths'][idx],
                distance=float(distances[0][i]),
                cluster_id=cluster_id,
                rank=i + 1,
                confidence=SimilarityMatcher._calculate_confidence(
                    distances[0][i], cluster_data)
            )
            results.append(result)

        return sorted(results, key=lambda x: x.distance)

    @staticmethod
    def expand_search(cluster_model: Any, cluster_index: Dict[int, Dict], query_features: np.ndarray,
                      top_k: int, max_clusters: int) -> List[SimilarityResult]:
        """
        扩展搜索到多个相关簇

        Args:
            cluster_model: 聚类模型
            cluster_index: 聚类索引
            query_features: 查询特征
            top_k: 每个簇的搜索数量
            max_clusters: 最大搜索簇数

        Returns:
            扩展搜索的结果列表
        """
        # 找到最相关的簇
        nearest_clusters = SimilarityMatcher._find_nearest_clusters(
            cluster_model, query_features, max_clusters)

        all_results = []
        for cluster_id, distance in nearest_clusters:
            cluster_results = SimilarityMatcher.search_in_cluster(
                cluster_index, cluster_id, query_features, top_k // max_clusters + 1)
            # 为跨簇搜索的结果添加距离惩罚
            for result in cluster_results:
                result.distance += distance * 0.1  # 簇中心距离的小幅惩罚
            all_results.extend(cluster_results)

        return all_results

    @staticmethod
    def merge_results(results1: List[SimilarityResult], results2: List[SimilarityResult],
                      max_results: int) -> List[SimilarityResult]:
        """
        合并两个搜索结果并去重

        Args:
            results1: 第一组结果
            results2: 第二组结果
            max_results: 最大结果数量

        Returns:
            合并后的结果列表
        """
        # 使用路径作为去重键
        path_to_result = {}

        # 合并结果，保留距离更小的
        for result in results1 + results2:
            if result.path not in path_to_result or result.distance < path_to_result[result.path].distance:
                path_to_result[result.path] = result

        # 排序并限制数量
        merged_results = list(path_to_result.values())
        merged_results.sort(key=lambda x: x.distance)

        return merged_results[:max_results]

    @staticmethod
    def _find_nearest_clusters(cluster_model: Any, query_features: np.ndarray, n_clusters: int) -> List[Tuple[int, float]]:
        """
        找到最接近查询特征的簇

        Args:
            cluster_model: 聚类模型
            query_features: 查询特征
            n_clusters: 返回的簇数量

        Returns:
            (簇ID, 到簇中心的距离)列表
        """
        if not hasattr(cluster_model, 'cluster_centers_'):
            return []

        centers = cluster_model.cluster_centers_.astype(np.float64)
        query_features_float64 = query_features.astype(np.float64)
        distances = np.linalg.norm(centers - query_features_float64, axis=1)

        nearest_indices = np.argsort(distances)[:n_clusters]

        return [(int(idx), float(distances[idx])) for idx in nearest_indices]

    @staticmethod
    def _calculate_confidence(distance: float, cluster_data: Dict) -> float:
        """
        通用置信度计算 - 混合自适应方法

        采用多策略融合的稳健算法，自动适应不同的数据分布特征：
        1. 完全匹配特殊处理 (distance=0 → confidence=1.0)
        2. 数据特征分析 (变异系数、偏度)
        3. 自适应方法选择 (指数衰减/归一化倒数/百分位数)
        4. 边界保护 (0.01 ≤ confidence ≤ 1.0)

        Args:
            distance: 到查询的距离
            cluster_data: 簇数据

        Returns:
            置信度分数 (0.01-1.0)
        """
        # 特殊处理：完全匹配的情况（自查询）
        if distance == 0:
            return 1.0

        # 数据验证
        cluster_features = cluster_data['features']
        if len(cluster_features) < 2:
            return 0.5

        # 计算簇内距离分布
        center = cluster_data['center']
        cluster_distances = np.linalg.norm(cluster_features - center, axis=1)

        # 分析数据特征
        mean_dist = np.mean(cluster_distances)
        std_dist = np.std(cluster_distances)

        if mean_dist <= 0:
            return 0.5

        # 计算变异系数和偏度
        cv = std_dist / mean_dist  # 变异系数

        # 根据数据特征选择最佳计算策略
        if cv < 0.3:  # 低变异 - 使用归一化倒数法
            max_dist = np.max(cluster_distances)
            if max_dist > 0:
                confidence = 1 / (1 + (distance / max_dist) + 0.1)
            else:
                confidence = 0.5

        elif std_dist > 0:  # 中等变异 - 使用混合方法
            # 方法1: 指数衰减 (主要策略)
            scale_factor = mean_dist
            exp_confidence = np.exp(-distance / scale_factor)

            # 方法2: 相对位置 (辅助策略)
            max_dist = np.max(cluster_distances)
            min_dist = np.min(cluster_distances)

            if max_dist > min_dist:
                rel_confidence = max(
                    0.1, 1 - (distance - min_dist) / (max_dist - min_dist))
            else:
                rel_confidence = 0.5

            # 加权融合 (指数衰减权重0.7，相对位置权重0.3)
            confidence = 0.7 * exp_confidence + 0.3 * rel_confidence

        else:  # 特殊情况
            confidence = 0.5

        # 边界保护和平滑处理
        confidence = max(0.01, min(1.0, confidence))

        return confidence

    @staticmethod
    def _filter_and_rank_results(results: List[SimilarityResult], query_path: str,
                                 top_k: int, threshold: float) -> List[SimilarityResult]:
        """
        过滤和重新排序结果

        Args:
            results: 原始结果列表
            query_path: 查询图片路径（用于过滤自己）
            top_k: 最终返回数量
            threshold: 相似性阈值 (0-1)
                - 0.05-0.1: 宽松模式，更多候选
                - 0.1-0.2: 平衡模式（推荐）
                - 0.2-0.3: 严格模式，高相似度
                - 0.3+: 极严格模式

        Returns:
            过滤后的结果列表
        """
        logger.info(f"开始过滤结果，原始结果数: {len(results)}, 阈值: {threshold}")

        # 过滤自身和低质量结果
        filtered = []
        distance_stats = []

        for result in results:
            # 跳过查询图片自身（可选）
            if result.path == query_path:
                continue

            distance_stats.append(result.distance)

            # 改进的相似度计算：适配实际距离范围的映射
            # 基于观察到的距离范围(8-120)，使用更合理的转换

            # 方案1：线性映射 + 归一化 (推荐)
            max_reasonable_distance = 120  # 基于观察到的最大距离
            similarity = max(0, 1 - result.distance / max_reasonable_distance)

            # 方案2：指数衰减 (适中的衰减率)
            # similarity = np.exp(-result.distance / 30)  # 30是调节参数

            # 方案3：反比例函数 (平缓的衰减)
            # similarity = 1 / (1 + result.distance / 20)  # 20是调节参数

            if similarity >= threshold:
                # 将计算出的相似度保存到结果中，便于调试
                result.similarity = similarity
                filtered.append(result)

        # 记录距离统计信息用于调试
        if distance_stats:
            logger.info(f"距离统计 - 最小: {min(distance_stats):.3f}, 最大: {max(distance_stats):.3f}, "
                        f"平均: {np.mean(distance_stats):.3f}, 中位数: {np.median(distance_stats):.3f}")

            # 计算相似度统计
            similarities = [np.exp(-d) for d in distance_stats]
            logger.info(f"相似度统计 - 最小: {min(similarities):.6f}, 最大: {max(similarities):.6f}, "
                        f"平均: {np.mean(similarities):.6f}")
            logger.info(f"满足阈值 {threshold} 的结果数: {len(filtered)}")
        else:
            logger.warning("没有找到任何结果进行过滤")

        # 重新排序：综合考虑距离和置信度
        for i, result in enumerate(filtered):
            # 综合分数：距离权重0.7，置信度权重0.3
            result.rank = i + 1

        # 排序策略：主要按距离，置信度作为微调
        filtered.sort(key=lambda x: x.distance - x.confidence * 0.1)

        logger.info(f"最终过滤结果数: {len(filtered)}")
        return filtered[:top_k]

    @staticmethod
    def find_similar_images_in_all_clusters(
        cluster_index: Dict[int, Dict],
        similarity_threshold: float = 0.9
    ) -> List[List[SimilarityResult]]:
        """
        查找所有簇内相似图片组（静态方法）

        这是一个高效的相似图片分组算法，利用现有聚类结构避免重复计算：
        1. 在每个簇内进行相似性分析
        2. 使用BallTree的radius_neighbors进行高效近邻查找
        3. 构建图结构并找到连通分量作为相似组
        4. 时间复杂度：O(N²/k) 其中k是簇数

        Args:
            cluster_index: 聚类索引
            similarity_threshold: 相似性阈值 (0-1)

        Returns:
            相似图片组列表，每组包含相似的图片结果
        """
        logger.info(f"开始在所有簇内查找相似图片组，阈值: {similarity_threshold}")

        all_groups = []
        total_processed = 0

        # 将相似度阈值转换为距离阈值
        # 基于观察到的距离范围(8-120)进行映射
        max_reasonable_distance = 120
        distance_threshold = (1 - similarity_threshold) * \
            max_reasonable_distance

        logger.info(
            f"相似度阈值 {similarity_threshold} 转换为距离阈值 {distance_threshold:.3f}")

        # 遍历每个簇
        for cluster_id, cluster_data in cluster_index.items():
            cluster_features = cluster_data['features']
            cluster_paths = cluster_data['paths']
            cluster_tree = cluster_data['tree']

            cluster_size = len(cluster_features)
            if cluster_size < 2:
                continue

            logger.info(f"处理簇 {cluster_id}，包含 {cluster_size} 张图片")

            # 在当前簇内查找相似图片组
            cluster_groups = SimilarityMatcher._find_groups_in_cluster(
                cluster_id, cluster_features, cluster_paths, cluster_tree,
                distance_threshold, cluster_data
            )

            all_groups.extend(cluster_groups)
            total_processed += cluster_size

            logger.info(f"簇 {cluster_id} 找到 {len(cluster_groups)} 个相似组")

        logger.info(f"总计处理 {total_processed} 张图片，找到 {len(all_groups)} 个相似组")

        return all_groups

    @staticmethod
    def _find_groups_in_cluster(
        cluster_id: int,
        features: np.ndarray,
        paths: List[str],
        tree: Any,
        distance_threshold: float,
        cluster_data: Dict
    ) -> List[List[SimilarityResult]]:
        """
        在单个簇内查找相似图片组

        使用图连通分量算法：
        1. 使用BallTree.query_radius找到每个点的近邻
        2. 构建邻接图
        3. 使用DFS找到所有连通分量

        Args:
            cluster_id: 簇ID
            features: 簇内特征矩阵
            paths: 图片路径列表
            tree: BallTree索引
            distance_threshold: 距离阈值
            cluster_data: 簇数据（用于置信度计算）

        Returns:
            该簇内的相似图片组列表
        """
        n_images = len(features)
        if n_images < 2:
            return []

        # 使用BallTree的query_radius方法高效查找近邻
        # 这比逐一query更高效，时间复杂度O(N log N)而非O(N²)
        neighbors_list = tree.query_radius(features, r=distance_threshold)

        # 构建邻接图（无向图）
        graph = [set() for _ in range(n_images)]
        for i, neighbors in enumerate(neighbors_list):
            for neighbor_idx in neighbors:
                if neighbor_idx != i:  # 排除自己
                    graph[i].add(neighbor_idx)
                    graph[neighbor_idx].add(i)  # 无向图

        # 使用DFS找到所有连通分量（相似组）
        visited = [False] * n_images
        groups = []

        for i in range(n_images):
            if not visited[i] and len(graph[i]) > 0:  # 未访问且有邻居
                # DFS遍历连通分量
                component = []
                stack = [i]

                while stack:
                    current = stack.pop()
                    if not visited[current]:
                        visited[current] = True
                        component.append(current)

                        # 添加未访问的邻居到栈中
                        for neighbor in graph[current]:
                            if not visited[neighbor]:
                                stack.append(neighbor)

                # 如果连通分量大小>=2，构建相似组
                if len(component) >= 2:
                    # 计算相似组（连通分量）的中心
                    component_features = features[component]
                    group_center = np.mean(component_features, axis=0)

                    group = []
                    for idx in component:
                        # 方案1：计算到相似组中心的距离（推荐）
                        distance = np.linalg.norm(features[idx] - group_center)

                        # 方案2：计算到组内其他图片的最小距离（更严格）
                        # min_distance = float('inf')
                        # for other_idx in component:
                        #     if other_idx != idx:
                        #         dist = np.linalg.norm(features[idx] - features[other_idx])
                        #         min_distance = min(min_distance, dist)
                        # distance = min_distance if min_distance != float('inf') else 0.0

                        # 方案3：计算到组内其他图片的平均距离
                        # if len(component) > 1:
                        #     distances_to_others = [
                        #         np.linalg.norm(features[idx] - features[other_idx])
                        #         for other_idx in component if other_idx != idx
                        #     ]
                        #     distance = np.mean(distances_to_others)
                        # else:
                        #     distance = 0.0

                        confidence = SimilarityMatcher._calculate_confidence(
                            distance, cluster_data)

                        # 计算相似度
                        max_reasonable_distance = 120
                        similarity = max(0, 1 - distance /
                                         max_reasonable_distance)

                        result = SimilarityResult(
                            path=paths[idx],
                            distance=float(distance),
                            cluster_id=cluster_id,
                            rank=len(group) + 1,
                            confidence=confidence,
                            similarity=similarity
                        )
                        group.append(result)

                    # 按距离排序组内图片
                    group.sort(key=lambda x: x.distance)

                    # 重新分配rank
                    for rank, result in enumerate(group):
                        result.rank = rank + 1

                    groups.append(group)

        return groups
