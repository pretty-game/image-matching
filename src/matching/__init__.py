"""
相似性匹配模块

实现基于聚类的图像相似性搜索
"""

from .similarity_matcher import SimilarityMatcher, SimilarityResult

__all__ = [
    'SimilarityMatcher',
    'SimilarityResult'
]