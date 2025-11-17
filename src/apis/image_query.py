# -*- coding: utf-8 -*-
"""
图片查询模块
从production_tools/image_query.py迁移过来的功能
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional

from src.main import GameAssetManager


class ImageQuery:
    """图片查询器"""

    def __init__(self, asset_manager: GameAssetManager):
        """
        初始化图片查询器
        Args:
            asset_manager: 游戏资产管理器实例
        """
        self.asset_manager = asset_manager
        self.logger = logging.getLogger("ImageQuery")

    def find_similar(self,
                    query_image_path: str,
                    top_k: int = 10,
                    similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        查找相似图片

        Args:
            query_image_path: 查询图片路径
            top_k: 返回结果数量
            similarity_threshold: 相似性阈值

        Returns:
            查询结果
        """
        result = {
            'success': False,
            'query_image': query_image_path,
            'similar_images': [],
            'execution_time': 0,
            'error': None
        }

        start_time = time.time()

        try:
            # 检查查询图片是否存在
            if not os.path.exists(query_image_path):
                result['error'] = f'查询图片不存在: {query_image_path}'
                return result

            # 检查系统是否已训练
            if not self.asset_manager.is_trained:
                result['error'] = '图片库未训练，请先调用update_library()'
                return result

            self.logger.info(f"查询相似图片: {os.path.basename(query_image_path)}")

            # 使用默认阈值
            if similarity_threshold is None:
                similarity_threshold = self.asset_manager.config.get('matching', {}).get('similarity_threshold', 0.9)

            # 执行查询
            similar_results = self.asset_manager.find_similar(query_image_path, top_k=top_k, similarity_threshold=similarity_threshold)

            # 格式化结果
            formatted_results = []
            for i, img_result in enumerate(similar_results):
                # 应用相似性阈值过滤
                if img_result.similarity < similarity_threshold:
                    continue

                formatted_result = {
                    'rank': i + 1,
                    'image_path': img_result.path,
                    'filename': os.path.basename(img_result.path),
                    'directory': os.path.dirname(img_result.path),
                    'similarity': img_result.similarity,
                    'confidence': img_result.confidence,
                    'distance': img_result.distance,
                    'file_size_mb': (
                        os.path.getsize(img_result.path) / (1024 * 1024)
                        if os.path.exists(img_result.path) else 0
                    )
                }
                formatted_results.append(formatted_result)

            result['similar_images'] = formatted_results
            result['success'] = True

            self.logger.info(f"找到 {len(formatted_results)} 张相似图片")

        except Exception as e:
            result['error'] = f'查询过程出错: {e}'
            self.logger.error(result['error'])

        finally:
            result['execution_time'] = time.time() - start_time

        return result
