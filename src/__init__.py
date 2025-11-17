"""
游戏资产图片聚类与匹配系统

主要模块导入
"""

from .main import GameAssetManager
from .feature_extraction import DCTFeatureExtractor, FeatureSelector
from .matching import SimilarityMatcher, SimilarityResult

__all__ = [
    'GameAssetManager',
    'DCTFeatureExtractor',
    'FeatureSelector',
    'SimilarityMatcher',
    'SimilarityResult'
]

__version__ = '1.0.0'