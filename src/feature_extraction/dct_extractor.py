"""
DCT特征提取模块

实现多通道DCT变换特征提取，专门优化处理RGBA游戏资产
"""
import hashlib
import logging
import os
from pathlib import Path
import pickle
import zlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.fft import dct

logger = logging.getLogger(__name__)

class DCTFeatureExtractor:
    """多通道DCT特征提取器，专门处理RGBA游戏资产图片"""

    @staticmethod
    def get_dct_features(image_paths: List[str], cache_dir: str, image_size: int, dct_size: int) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """
        提取一组图片的DCT特征，支持缓存机制
        Args:
            image_paths: 图片路径列表
            cache_dir: 缓存目录
            image_size: 输入图像尺寸（用于验证尺寸一致性）
            dct_size: DCT低频分量提取尺寸
        Returns:
            Tuple containing:
                - features: numpy.ndarray of extracted DCT features
                - paths: List of image paths corresponding to the features
                - file_last_modified_times: List of last modified timestamps for each image file
                - file_crc32_list: List of CRC32 checksums for each image file
        """

        features_list = []
        path_list = []
        file_last_modified_times = []
        file_crc32_list = []
        cache_counter = 0
        for idx, img_path in enumerate(image_paths):
            logger.info(f"处理图片 {idx+1}/{len(image_paths)}: {img_path}")
            dct_cache_data, from_cache = DCTFeatureExtractor.get_dct_data(img_path, cache_dir, image_size, dct_size)
            if dct_cache_data is not None:
                features_list.append(dct_cache_data['dct_features'])
                path_list.append(img_path)
                file_last_modified_times.append(dct_cache_data['file_last_modified'])
                file_crc32_list.append(dct_cache_data['file_crc32'])
                if from_cache:
                    cache_counter += 1

        if not features_list:
            logger.info("没有成功提取到任何DCT特征, 请检查图片文件是否有效")
            return np.empty((0, dct_size * dct_size * 4), dtype=np.float32), [], [], []

        features = np.array(features_list)

        logger.info(f"DCT特征缓存命中率: {cache_counter}/{len(path_list)}")
        logger.info(f"DCT特征准备完成，有效图片: {len(path_list)}, 特征形状: {features.shape}")
        return features, path_list, file_last_modified_times, file_crc32_list

    @staticmethod
    def get_dct_data(image_path: str, cache_dir: str, image_size: int, dct_size: int) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        获取图片的DCT特征，优先从缓存加载

        Args:
            image_path: 图片路径
            cache_dir: 缓存目录
            image_size: 输入图像尺寸（用于验证尺寸一致性）
            dct_size: DCT低频分量提取尺寸
        Returns:
            包含特征和元数据的字典
        """
        # 尝试从缓存加载
        cached_data = DCTFeatureExtractor.load_dct_data(
            file_path=image_path,
            cache_dir=cache_dir,
            image_size=image_size,
            dct_size=dct_size
        )
        if cached_data is not None:
            # logger.info(f"DCT特征从缓存加载: {image_path}")
            return cached_data, True

        # 缓存不存在或失效，重新提取
        dct_data = DCTFeatureExtractor.extract_multi_channel_dct(
            image_path=image_path,
            image_size=image_size,
            dct_size=dct_size
        )
        if dct_data is not None:
            # logger.info(f"DCT特征提取成功: {image_path}")

            # 保存到缓存
            DCTFeatureExtractor.save_dct_data(
                file_path=image_path,
                cache_dir=cache_dir,
                dct_feature={
                    'dct_features': dct_data['dct_features'],
                    'image_size': image_size,
                    'dct_size': dct_size,
                    'file_last_modified': os.path.getmtime(image_path),
                    'file_crc32': dct_data['file_crc32'],
                }
            )
        return dct_data, False

    @staticmethod
    def load_image(image_path: str, image_size: int, code: int) -> Optional[np.ndarray]:
        """
        快速加载图片并转换为指定颜色空间+A格式（支持中文路径）
        返回 (H, W, 4)，通道顺序[code channels..., A]
        """

        # logger.info(f"开始加载图片: {image_path}")

        try:
            # 使用 numpy 和 cv2.imdecode 支持中文路径
            with open(image_path, 'rb') as f:
                file_data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_UNCHANGED)

            if img is None:
                logger.warning(f"无法读取图片: {image_path}")
                return None

            # 转换为指定颜色空间+A格式
            img_converted = DCTFeatureExtractor._convert_to(img, code)
            # 调整尺寸
            img_resized = DCTFeatureExtractor._resize_without_padding(img_converted, image_size)
            # 归一化到 [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            return img_normalized
        except Exception as e:
            logger.warning(f"跳过图片 {os.path.basename(image_path)}: {e}")
            return None

    @staticmethod
    def _convert_to(img: np.ndarray, code: int) -> np.ndarray:
        """
        将任意格式图片直接转换为YCrCb+A格式
        返回 (H, W, 4)，通道顺序[Y, Cr, Cb, A]
        """
        if len(img.shape) == 2:
            # 灰度图，转为BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = np.full(img.shape[:2], 255, dtype=img.dtype)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # BGR
                alpha = np.full(img.shape[:2], 255, dtype=img.dtype)
            elif img.shape[2] == 4:
                # BGRA
                alpha = img[:, :, 3]
                img = img[:, :, :3]
            else:
                raise ValueError("不支持的通道数")
        else:
            raise ValueError("不支持的图片格式")

        # BGR -> code
        converted = cv2.cvtColor(img, code)
        # 合并code和A通道
        result = np.dstack([converted, alpha])
        return result

    @staticmethod
    def load_rgba_image(image_path: str, image_size: int) -> Optional[np.ndarray]:
        """
        快速加载RGBA图片（支持中文路径）
        """

        # logger.info(f"开始加载图片: {image_path}")

        try:
            # 使用 numpy 和 cv2.imdecode 支持中文路径
            with open(image_path, 'rb') as f:
                file_data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_UNCHANGED)

            if img is None:
                logger.warning(f"无法读取图片: {image_path}")
                return None

            # 转换为RGBA格式
            img_rgba = DCTFeatureExtractor._convert_to_rgba(img)
            # 调整尺寸
            img_resized = DCTFeatureExtractor._resize_without_padding(img_rgba, image_size)
            # 归一化到 [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            return img_normalized
        except Exception as e:
            logger.warning(f"跳过图片 {os.path.basename(image_path)}: {e}")
            return None

    @staticmethod
    def _convert_to_rgba(img: np.ndarray) -> np.ndarray:
        """
        将任何格式图片转换为RGBA
        """
        if len(img.shape) == 2:
            # 灰度 -> RGBA
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # BGR -> RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            elif img.shape[2] == 4:
                # BGRA -> RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        return img

    @staticmethod
    def _resize_with_padding(img: np.ndarray, image_size: int) -> np.ndarray:
        """
        剔除边缘透明像素后，调整图片大小并居中填充
        """

        # 剔除边缘透明像素
        alpha_channel = img[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

        if img.size == 0:
            # 全透明图片，返回空白画布
            return np.zeros((image_size, image_size, 4), dtype=img.dtype)

        h, w = img.shape[:2]
        target_w, target_h = image_size, image_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)

        # 新尺寸
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # 缩放
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建目标画布（透明背景）
        result = np.zeros((target_h, target_w, 4), dtype=img.dtype)

        # 居中放置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        return result

    @staticmethod
    def _resize_without_padding(img: np.ndarray, image_size: int) -> np.ndarray:
        """
        直接调整图片大小，不进行填充
        """
        img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        return img_resized

    @staticmethod
    def extract_multi_channel_dct(image_path: str, image_size: int, dct_size: int) -> Optional[Dict[str, Any]]:
        """
        提取四个通道的DCT特征，包含元数据，完成后缓存
        Args:
            image_path: 图片路径
            image_size: 输入图像尺寸（用于验证尺寸一致性）
            dct_size: DCT低频分量提取尺寸
        Returns:
            包含特征和元数据的字典
        """

        # image_four_channel = DCTFeatureExtractor.load_rgba_image(image_path, image_size)
        image_four_channel = DCTFeatureExtractor.load_image(image_path, image_size, cv2.COLOR_BGR2Lab)
        if image_four_channel is None:
            return None
        # 提取每个通道的DCT特征
        channel_features = []
        raw_dct_features = np.zeros((4, dct_size, dct_size))

        for channel_idx in range(4):  # L, a, b, A
            channel_data = image_four_channel[:, :, channel_idx]
            dct_features = DCTFeatureExtractor.calculate_dct(channel_data, image_size, dct_size)
            raw_dct_features[channel_idx] = dct_features
            channel_features.append(dct_features.flatten())

        # 拼接所有通道特征
        combined_features = np.concatenate(channel_features)

        file_crc32 = 0
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                file_crc32 = zlib.crc32(chunk, file_crc32)
        file_crc32 = hex(file_crc32 & 0xffffffff)

        return {
            'file_path': image_path,
            'file_last_modified': os.path.getmtime(image_path),
            'file_crc32': file_crc32,
            'image_size': image_size,
            'dct_size': dct_size,
            'dct_features': np.array(combined_features, dtype=np.float32),
        }

    @staticmethod
    def calculate_dct(image_data: np.ndarray, image_size: int, dct_size: int) -> np.ndarray:
        """
        计算单通道图像的DCT特征

        Args:
            image_data: 单通道图像数据 (H, W) - 应该已经是image_size尺寸
            image_size: 输入图像尺寸（用于验证尺寸一致性）

        Returns:
            DCT特征 (dct_size, dct_size) - 低频分量
        """
        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)
        if image_data.shape != (image_size, image_size):
            logger.warning(f"图片尺寸不匹配 {image_data.shape} != ({image_size}, {image_size})，执行resize")
            image_data = DCTFeatureExtractor._resize_without_padding(image_data, image_size)  # type: ignore

        # 二维DCT变换
        dct_result = dct(dct(image_data, axis=0, norm='ortho'), axis=1, norm='ortho')
        # 取左上角低频部分
        dct_features = dct_result[:dct_size, :dct_size]

        return dct_features

    @staticmethod
    def save_dct_data(file_path: str, cache_dir: str, dct_feature: Dict[str, Any]) -> bool:
        """保存DCT特征到缓存"""
        try:
            cache_path = DCTFeatureExtractor._get_file_path_cache_path(file_path, cache_dir)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            with open(cache_path, 'wb') as f:
                pickle.dump(dct_feature, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug(f"DCT特征已缓存: {file_path} -> {cache_path}")
            return True

        except Exception as e:
            logger.error(f"保存DCT特征失败 {file_path}: {e}")
            return False

    @staticmethod
    def load_dct_data(file_path: str, cache_dir: str, image_size: int, dct_size: int) -> Optional[Dict[str, Any]]:
        """
        从缓存加载DCT特征，支持配置变更检测

        Args:
            file_path: 原始文件路径
            image_size: 当前图像尺寸配置
            dct_size: 当前DCT尺寸配置
        Returns:
            包含DCT特征的字典或None（如果缓存失效）
            返回格式: {
                'dct_features': np.ndarray
            }
        """
        try:
            cache_path = DCTFeatureExtractor._get_file_path_cache_path(file_path, cache_dir)
            if not os.path.exists(cache_path):
                return None
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            # 检查缓存格式兼容性 - 如果不包含必要字段则视为不兼容
            if 'dct_features' not in data:
                logger.debug(f"缓存格式不兼容（缺少dct_features字段），失效: {file_path}")
                return None
            if 'image_size' not in data:
                logger.debug(f"缓存格式不兼容（缺少image_size字段），失效: {file_path}")
                return None
            if data['image_size'] != image_size:
                logger.debug(f"配置{image_size}变更，缓存失效: {file_path}")
                return None
            if 'dct_size' not in data:
                logger.debug(f"缓存格式不兼容（缺少dct_size字段），失效: {file_path}")
                return None
            if data['dct_size'] != dct_size:
                logger.debug(f"配置{dct_size}变更，缓存失效: {file_path}")
                return None
            if 'file_crc32' not in data:
                logger.debug(f"缓存格式不兼容（缺少file_crc32字段），失效: {file_path}")
                return None
            # 先判断文件修改时间
            if 'file_last_modified' in data:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    if data['file_last_modified'] == file_mtime:
                        logger.debug(f"DCT特征从缓存加载(仅mtime): {file_path}")
                        return data
                except Exception as e:
                    logger.debug(f"获取文件修改时间失败: {file_path}, {e}")
            # 修改时间不同再判断crc32
            file_crc32 = 0
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    file_crc32 = zlib.crc32(chunk, file_crc32)
            file_crc32 = hex(file_crc32 & 0xffffffff)
            if data['file_crc32'] != file_crc32:
                logger.debug(f"文件内容变更，缓存失效: {file_path}")
                return None
            logger.debug(f"DCT特征从缓存加载(通过crc32): {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载DCT特征失败 {file_path}: {e}")
            return None

    @staticmethod
    def _get_file_path_cache_path(file_path: str, cache_dir: str) -> str:
        """根据文件路径生成缓存文件路径"""
        # 使用文件路径的hash作为缓存文件名，避免路径长度和特殊字符问题
        file_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, 'dct_cache', f"{file_hash}.pkl")
        return cache_path