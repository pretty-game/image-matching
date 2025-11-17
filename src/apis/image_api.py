# -*- coding: utf-8 -*-
"""
图片匹配系统统一API接口
简化的对外接口，整合所有核心功能
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from src.main import GameAssetManager
from src.apis.report_generator import ReportGenerator
from src.apis.image_query import ImageQuery

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ImageMatchingAPI:
    """
    图片匹配系统统一API接口
    提供简化的对外接口，隐藏内部复杂性
    """

    def __init__(self, config_path: str = "./config/settings.yaml"):
        """
        初始化图片匹配API

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.asset_manager = None
        self.report_generator = None
        self.image_query = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("ImageMatchingAPI")

        # 避免重复添加 handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            # 防止日志传播到根 logger 造成重复输出
            logger.propagate = False

        return logger

    def initialize(self) -> bool:
        """
        初始化系统

        Returns:
            是否初始化成功
        """
        try:
            self.logger.info("正在初始化图片匹配系统...")

            # 初始化核心管理器
            self.asset_manager = GameAssetManager(self.config_path)
            self.asset_manager.load_clustering_system()

            # 初始化子模块
            self.report_generator = ReportGenerator(self.asset_manager)
            self.image_query = ImageQuery(self.asset_manager)

            self.logger.info("图片匹配系统初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False

    def generate_similarity_report(
        self,
        output_path: str = "reports/similarity_report.html",
        similarity_threshold: float = 0.9,
        min_group_size: int = 2,
        format_type: str = "html",
        max_groups: int = 5000,
        lightweight: bool = False,
        use_optimized: bool = True
    ) -> Dict[str, Any]:
        """
        生成相似图片报告

        Args:
            output_path: 输出文件路径
            similarity_threshold: 相似性阈值
            min_group_size: 最小组大小
            format_type: 报告格式 (html, json)
            max_groups: 最大组数限制（用于性能优化）
            lightweight: 轻量级模式（不生成缩略图，速度更快）
            use_optimized: 是否使用优化算法（基于聚类的高效方法）

        Returns:
            生成结果
        """
        if not self.report_generator:
            raise ValueError("系统未初始化，请先调用initialize()")

        return self.report_generator.generate_report(
            output_path=output_path,
            similarity_threshold=similarity_threshold,
            min_group_size=min_group_size,
            format_type=format_type,
            max_groups=max_groups,
            lightweight=lightweight,
            use_optimized=use_optimized
        )

    def find_similar_images(
        self,
        query_image_path: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        查找相似图片

        Args:
            query_image_path: 查询图片路径
            top_k: 返回结果数量
            similarity_threshold: 相似性阈值

        Returns:
            查询结果
        """
        if not self.image_query:
            raise ValueError("系统未初始化，请先调用initialize()")

        return self.image_query.find_similar(
            query_image_path=query_image_path,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    def health_check(self) -> Dict[str, Any]:
        """
        系统健康检查

        Returns:
            健康检查结果
        """
        try:
            # 检查配置文件
            config_exists = os.path.exists(self.config_path)

            # 检查缓存目录
            cache_exists = os.path.exists("cache")

            # 检查是否已训练
            trained = self.asset_manager.is_trained if self.asset_manager else False

            return {
                'status': 'healthy' if all([config_exists, cache_exists]) else 'unhealthy',
                'config_exists': config_exists,
                'cache_exists': cache_exists,
                'trained': trained,
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
