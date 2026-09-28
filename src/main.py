"""
游戏资产管理主控制器

整合预处理、特征提取、聚类和相似性匹配的完整流程
"""
import hashlib
import logging
import os
from pathlib import Path
import pickle
from datetime import datetime
from typing import Any, Dict, List, Tuple
import zlib

import yaml

from src.feature_extraction.dct_extractor import DCTFeatureExtractor
from src.feature_extraction.feature_selector import FeatureSelector

from .clustering import GameAssetClusterer
from .matching import SimilarityMatcher, SimilarityResult


def setup_logging():
    """设置日志配置，输出到cache目录并使用时间戳作为文件名"""
    # 创建cache目录
    cache_dir = Path('./log')
    cache_dir.mkdir(exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = cache_dir / f"log_{timestamp}.log"

    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除可能存在的默认处理器
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_filename


# 日志初始化由 app.py 负责，此处不自动调用
logger = logging.getLogger(__name__)


class GameAssetManager:
    """游戏资产管理主控制器"""

    def __init__(self, config_path: str = './config/settings.yaml'):
        """
        初始化资产管理器

        Args:
            config_path: 配置文件路径
        """

        # 初始化配置
        self.config_path = config_path
        self.config = GameAssetManager.load_config(config_path)

        self.asset_directories = self.config.get(
            'data_sources', {}).get('asset_directories', [])
        self.asset_directories.sort()

        feature_extraction_config = self.config['feature_extraction']
        self.image_size = feature_extraction_config.get('image_size', 256)
        self.dct_size = feature_extraction_config.get('dct_size', 64)
        self.feature_dim = feature_extraction_config.get('feature_dim', 500)

        # 存储聚类参数
        clustering_config = self.config['clustering']
        self.n_clusters = clustering_config['n_clusters']
        self.batch_size = clustering_config['batch_size']

        # 有效的图片路径列表
        self.valid_image_paths = []
        self.valid_image_paths_last_modified = []
        self.valid_image_paths_crc32 = []

        # 存储特征选择器的索引，用于静态方法调用
        self.selected_info = None
        self.selected_features = None

        # 存储聚类结果
        self.cluster_labels = None
        self.kmeans_model = None
        self.cluster_index = None

        # 状态变量
        self.is_trained = False

        # 变更记录
        self.change_record = None

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {config_path}，使用默认配置")
            raise FileNotFoundError
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}，使用默认配置")
            raise Exception

    @staticmethod
    def get_image_paths(directories: List[str], extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tga')) -> List[str]:
        """
        扫描目录中的图片文件

        Args:
            directories: 目录路径列表
            extensions: 支持的文件扩展名

        Returns:
            图片文件路径列表
        """
        image_paths = []

        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(extensions):
                        image_paths.append(os.path.join(root, file))

        logger.info(f"在 {len(directories)} 个目录中找到 {len(image_paths)} 个图片文件")
        return image_paths

    # region Path

    @staticmethod
    def get_directories_hash(directories: List[str]) -> str:
        """
        缓存图库 file_crc32 状态，用于后续增量更新

        Args:
            directories: 目录路径列表
            cache_dir: 缓存目录路径
        """

        directories.sort()
        directories_hash = hashlib.md5(
            "".join(directories).encode()).hexdigest()
        return directories_hash

    # endregion

    # region Cache

    @staticmethod
    def resolve_cache_dir(cache_dir: str, config_path: str) -> str:
        """
        解析缓存目录：相对路径以项目根目录（config 文件上一级）为基准，
        避免受进程启动位置（CWD）影响。例如从 webapp/ 目录启动服务时，
        './cache' 不会被错误解析到 webapp/cache 下。
        """
        p = Path(cache_dir)
        if not p.is_absolute():
            project_root = Path(config_path).resolve().parent.parent
            p = project_root / p
        return str(p)

    @staticmethod
    def get_change_record(cache_file_paths: List[str], cache_file_last_modified_times: List[str], cache_file_crc32s: List[str], current_file_paths: List[str]) -> Dict[str, List[str]]:
        """
        根据缓存的 file_crc32 状态，返回增删改列表
        优化版本：先用文件修改时间快速过滤，只对可能变化的文件计算CRC32

        Args:
            cache_file_paths: 缓存的文件路径列表
            cache_file_last_modified_times: 缓存的文件最后修改时间列表
            cache_file_crc32s: 缓存的文件 CRC32 列表
            current_file_paths: 当前的文件路径列表

        Returns:
            增删改记录
        """

        ret = {
            "added_image_paths": [],
            "deleted_image_paths": [],
            "modified_image_paths": []
        }

        # 构建缓存字典
        cache_file_dict = dict(zip(cache_file_paths, cache_file_crc32s))
        cache_mtime_dict = dict(
            zip(cache_file_paths, cache_file_last_modified_times))
        cache_file_set = set(cache_file_paths)
        current_file_set = set(current_file_paths)

        # 1. 快速识别新增和删除
        added_files = current_file_set - cache_file_set
        deleted_files = cache_file_set - current_file_set

        ret['added_image_paths'] = list(added_files)
        ret['deleted_image_paths'] = list(deleted_files)

        # 2. 对于同时存在的文件，先用修改时间快速判断
        common_files = cache_file_set & current_file_set

        # 检查可能被修改的文件（修改时间不同，保持float精度）
        potentially_modified = []
        for image_path in common_files:
            try:
                stat = os.stat(image_path)
                current_mtime = stat.st_mtime  # 保持float类型
                cached_mtime_value = cache_mtime_dict.get(image_path)

                # 统一转换为float类型进行比较
                cached_mtime = float(cached_mtime_value) if cached_mtime_value is not None else -1.0

                # 如果修改时间不同，标记为可能修改
                if cached_mtime < 0 or current_mtime != cached_mtime:
                    potentially_modified.append(image_path)
            except (OSError, IOError):
                # 文件访问失败，标记为可能修改
                potentially_modified.append(image_path)

        # 3. 只对可能被修改的文件计算CRC32验证
        if potentially_modified:
            logger.info(
                f"检测到 {len(potentially_modified)} 个文件修改时间变化，验证CRC32...")
            for image_path in potentially_modified:
                file_crc32 = 0
                try:
                    with open(image_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            file_crc32 = zlib.crc32(chunk, file_crc32)
                    file_crc32 = hex(file_crc32 & 0xffffffff)

                    if cache_file_dict[image_path] != file_crc32:
                        ret['modified_image_paths'].append(image_path)
                except Exception as e:
                    logger.warning(f"无法读取文件 {image_path}: {e}")

        logger.info(f"变更检测完成 - 新增: {len(ret['added_image_paths'])}, "
                    f"删除: {len(ret['deleted_image_paths'])}, "
                    f"修改: {len(ret['modified_image_paths'])}")

        return ret

    # endregion

    # region User API

    def load_clustering_system(self):
        """
        加载聚类系统
        """

        # 1. 遍历图片文件
        logger.info("步骤1: 遍历图片文件...")
        image_paths = GameAssetManager.get_image_paths(self.asset_directories)
        if not image_paths:
            raise ValueError(f"未找到有效的图片文件，请检查目录 {self.asset_directories} 是否包含图片文件")

        # 2. 尝试加载索引缓存
        logger.info("步骤2: 尝试加载索引缓存...")
        cache_dir = GameAssetManager.resolve_cache_dir(
            self.config.get('performance', {}).get('cache_dir', './cache'),
            self.config_path)
        index_data = None
        directories_hash = GameAssetManager.get_directories_hash(
            self.asset_directories)
        filepath = os.path.join(cache_dir, f'index_{directories_hash}.pkl')
        if not os.path.exists(filepath):
            logger.info(f"特征选择器文件不存在: {filepath}")
        else:
            try:
                with open(filepath, 'rb') as f:
                    index_data = pickle.load(f)
                logger.info(f"特征选择器已从 {filepath} 加载")
            except Exception as e:
                logger.warning(f"加载特征选择器失败 {filepath}: {e}")

        index_valid = False
        if index_data is not None:
            index_valid = True
            cache_meta = index_data.get('meta', {})
            if set(cache_meta.get('directories', [])) != set(self.asset_directories):
                logger.info("图库目录列表与缓存不匹配，缓存无效")
                index_valid = False
            if cache_meta.get('image_size') != self.image_size:
                logger.info("图片尺寸与缓存不匹配，缓存无效")
                index_valid = False
            if cache_meta.get('dct_size') != self.dct_size:
                logger.info("DCT尺寸与缓存不匹配，缓存无效")
                index_valid = False
            if cache_meta.get('feature_dim') != self.feature_dim:
                logger.info("特征选择器维度不匹配，缓存无效")
                index_valid = False
            if cache_meta.get('n_clusters') != self.n_clusters:
                logger.info("聚类数量不匹配，缓存无效")
                index_valid = False
            if cache_meta.get('batch_size') != self.batch_size:
                logger.info("聚类批处理大小不匹配，缓存无效")
                index_valid = False

        is_rebuild_selector = True
        is_rebuild_cluster = True
        if index_valid:
            valid_paths = index_data['valid_paths']
            file_last_modified_times = index_data['file_last_modified_times']
            file_crc32s = index_data['file_crc32s']
            selected_features = index_data['selected_features']
            selected_info = index_data['selected_info']
            kmeans_model = index_data['kmeans_model']
            cluster_labels = index_data['cluster_labels']
            change_record = GameAssetManager.get_change_record(
                valid_paths,
                file_last_modified_times,
                file_crc32s,
                image_paths
            )

            change_count = (len(change_record['added_image_paths']) +
                            len(change_record['deleted_image_paths']) +
                            len(change_record['modified_image_paths']))

            if change_count > 0:
                modify_ratio = change_count / len(image_paths)
                if modify_ratio < 0.1:
                    if GameAssetClusterer.supports_incremental_update(kmeans_model):
                        logger.info(f"累积变更比例较低（{modify_ratio:.2%}），尝试增量更新聚类系统")
                        all_deleted_image_paths = change_record['deleted_image_paths'] + \
                            change_record['modified_image_paths']
                        all_added_image_paths = change_record['added_image_paths'] + \
                            change_record['modified_image_paths']
                        added_dct_features, added_valid_paths, _, _ = DCTFeatureExtractor.get_dct_features(
                            all_added_image_paths, cache_dir, self.image_size, self.dct_size)
                        if len(added_valid_paths) > 0:
                            added_dct_selected_features = FeatureSelector.transform_features(
                                added_dct_features, selected_info)
                            cluster_labels, kmeans_model, selected_features, valid_paths = GameAssetClusterer.update_clusterer_incremental(
                                kmeans_model,
                                cluster_labels,
                                valid_paths,
                                selected_features=selected_features,
                                deleted_image_paths=all_deleted_image_paths,
                                added_image_paths=added_valid_paths,
                                added_features=added_dct_selected_features
                            )
                        else:
                            logger.info(f"增量更新时没有有效新增图片，跳过增量更新")
                        is_rebuild_selector = False
                        is_rebuild_cluster = False
                    else:
                        logger.info(f"小样本聚类模型不支持增量更新，重新构建聚类模型")
                        is_rebuild_selector = False
                else:
                    logger.info(
                        f"累积变更比例较高（{modify_ratio:.2%}），需要重新构建特征选择器和聚类模型")
            else:
                logger.info(f"缓存索引有效，无图片变更，加载完成")
                is_rebuild_selector = False
                is_rebuild_cluster = False

        if is_rebuild_selector:
            dct_features, valid_paths, file_last_modified_times, file_crc32s = DCTFeatureExtractor.get_dct_features(
                image_paths, cache_dir, self.image_size, self.dct_size)
            logger.info("重新构建索引缓存")
            selected_features, selected_info = FeatureSelector.select_important_features(
                dct_features, self.feature_dim)
            cluster_labels, kmeans_model = GameAssetClusterer.cluster_assets(
                selected_features, self.n_clusters, self.batch_size)
            logger.info("索引缓存构建完成")
        elif is_rebuild_cluster:
            logger.info("重新构建聚类模型")
            cluster_labels, kmeans_model = GameAssetClusterer.cluster_assets(
                selected_features, self.n_clusters, self.batch_size)
            logger.info("聚类模型构建完成")

        if is_rebuild_selector or is_rebuild_cluster:
            # 缓存索引
            index_data = {
                'meta': {
                    'directories': self.asset_directories,
                    'image_size': self.image_size,
                    'dct_size': self.dct_size,
                    'feature_dim': self.feature_dim,
                    'n_clusters': self.n_clusters,
                    'batch_size': self.batch_size
                },
                'valid_paths': valid_paths,
                'file_last_modified_times': file_last_modified_times,
                'file_crc32s': file_crc32s,
                'selected_features': selected_features,
                'selected_info': selected_info,
                'kmeans_model': kmeans_model,
                'cluster_labels': cluster_labels
            }

            logger.info(f"将索引缓存保存到 {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(index_data, f)

        self.valid_image_paths = valid_paths
        self.valid_image_paths_last_modified = file_last_modified_times
        self.valid_image_paths_crc32 = file_crc32s

        self.selected_features = selected_features
        self.selected_info = selected_info

        self.kmeans_model = kmeans_model
        self.cluster_labels = cluster_labels
        self.cluster_index = GameAssetClusterer.build_cluster_index(
            cluster_labels,
            valid_paths,
            selected_features
        )

        self.is_trained = True

    def find_similar(self, query_image_path: str, top_k: int, similarity_threshold: float) -> List[SimilarityResult]:
        """
        查找相似图片

        Args:
            query_image_path: 查询图片路径
            top_k: 返回数量，默认使用配置中的值
            similarity_threshold: 相似性阈值

        Returns:
            相似图片结果列表
        """

        if not self.is_trained:
            raise ValueError("系统尚未训练，请先调用 load_clustering_system()")

        if top_k is None:
            top_k = self.config['matching']['default_top_k']

        return SimilarityMatcher.find_similar_images(
            self.kmeans_model,
            self.cluster_index,
            self.selected_info,
            query_image_path,
            image_size=self.image_size,
            dct_size=self.dct_size,
            top_k=top_k,
            max_clusters=self.config['matching']['max_search_clusters'],
            similarity_threshold=similarity_threshold
        )

    # endregion
