# -*- coding: utf-8 -*-
"""
报告生成器模块
从production_tools/similarity_grouper.py迁移过来的功能
"""

import os
import io
import time
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np


class ReportGenerator:
    """报告生成器"""

    def __init__(self, asset_manager):
        self.asset_manager = asset_manager
        self.logger = logging.getLogger("ReportGenerator")

    def generate_report(self,
                       output_path: str,
                       similarity_threshold: float = 0.9,
                       min_group_size: int = 2,
                       format_type: str = "html",
                       max_groups: int = 5000,
                       lightweight: bool = False,
                       use_optimized: bool = True) -> Dict[str, Any]:
        """
        生成相似图片报告

        Args:
            output_path: 输出路径
            similarity_threshold: 相似性阈值
            min_group_size: 最小组大小
            format_type: 报告格式
            max_groups: 最大组数限制（用于性能优化）
            lightweight: 轻量级模式（不生成缩略图）
            use_optimized: 是否使用优化算法（基于聚类的高效方法）

        Returns:
            lightweight: 轻量级模式（不生成缩略图）

        Returns:
            生成结果
        """
        if not self.asset_manager.is_trained:
            return {
                'success': False,
                'error': '图片库未训练，请先调用update_library()'
            }

        start_time = time.time()

        try:
            # 自动生成带时间戳的文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 如果输出路径只指定了目录，使用默认文件名格式
            if os.path.isdir(output_path) or output_path.endswith('/') or output_path.endswith('\\'):
                output_dir = output_path
                timestamped_output = os.path.join(output_dir, f"report_{timestamp}.html")
            else:
                # 如果指定了文件名，在reports目录下使用report_时间戳格式
                output_dir = os.path.dirname(output_path) or "reports"
                timestamped_output = os.path.join(output_dir, f"report_{timestamp}.html")

            self.logger.info(f"开始生成相似性报告，阈值: {similarity_threshold}")

            # 选择分组算法：优化版本或原始版本
            if use_optimized:
                self.logger.info("使用优化的聚类分组算法")
                groups = self._find_similarity_groups_optimized(similarity_threshold, min_group_size)
            else:
                self.logger.info("使用原始的遍历分组算法")
                groups = self._find_similarity_groups(similarity_threshold, min_group_size)

            # 如果设置了最大组数限制，进行截取
            if max_groups and len(groups) > max_groups:
                self.logger.warning(f"相似组数量 {len(groups)} 超过限制 {max_groups}，将截取前 {max_groups} 组")
                groups = groups[:max_groups]

            # 根据格式生成报告
            if format_type.lower() == "html":
                self._generate_html_report(groups, timestamped_output, similarity_threshold, lightweight)
            elif format_type.lower() == "json":
                import json
                os.makedirs(os.path.dirname(timestamped_output), exist_ok=True)
                json_output = timestamped_output[:-5] + ".json" if timestamped_output.endswith('.html') else timestamped_output + ".json"

                # 创建包含元数据的JSON报告
                report_data = {
                    'metadata': {
                        'similarity_threshold': similarity_threshold,
                        'min_group_size': min_group_size,
                        'total_groups': len(groups),
                        'total_similar_images': sum(len(group) for group in groups),
                        'max_group_size': max(len(group) for group in groups) if groups else 0,
                        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'groups': groups
                }

                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
                timestamped_output = json_output
            else:
                return {
                    'success': False,
                    'error': f'不支持的格式: {format_type}'
                }

            total_similar_images = sum(len(group) for group in groups)

            return {
                'success': True,
                'output_path': timestamped_output,
                'total_groups': len(groups),
                'total_similar_images': total_similar_images,
                'similarity_ratio': total_similar_images / len(self.asset_manager.valid_image_paths) if self.asset_manager.valid_image_paths else 0,
                'execution_time': time.time() - start_time
            }

        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _find_similarity_groups(self, threshold: float, min_size: int) -> List[List[Dict[str, Any]]]:
        """简化的相似性分组算法"""
        groups = []
        processed_images = set()

        # 遍历所有图片，寻找相似组
        for image_path in self.asset_manager.valid_image_paths:
            if image_path in processed_images:
                continue

            # 查找当前图片的相似图片
            similar_results = self.asset_manager.find_similar(image_path, top_k=50)

            # 过滤出满足阈值的图片
            similar_group = [{'image_path': image_path, 'similarity': 1.0, 'confidence': 1.0}]
            for result in similar_results:
                if result.similarity >= threshold and result.path not in processed_images:
                    similar_group.append({
                        'image_path': result.path,
                        'similarity': result.similarity,
                        'confidence': getattr(result, 'confidence', 0.95)
                    })

            # 如果组大小满足要求，加入结果
            if len(similar_group) >= min_size:
                groups.append(similar_group)
                for item in similar_group:
                    processed_images.add(item['image_path'])

        self.logger.info(f"找到 {len(groups)} 个相似组")
        return groups

    def _find_similarity_groups_optimized(self, similarity_threshold: float, min_size: int) -> List[List[Dict[str, Any]]]:
        """
        优化的相似性分组算法 - 基于聚类的高效方法

        利用现有聚类结构，时间复杂度从O(N²)优化到O(N²/k)，其中k是簇数
        预期性能提升：k倍（例如100个簇可提升100倍性能）

        Args:
            similarity_threshold: 相似性阈值
            min_size: 最小组大小

        Returns:
            相似图片组列表
        """
        from src.matching.similarity_matcher import SimilarityMatcher

        self.logger.info(f"使用优化算法查找相似组，阈值: {similarity_threshold}, 最小组大小: {min_size}")

        # 检查是否有聚类数据
        if not hasattr(self.asset_manager, 'cluster_index') or not self.asset_manager.cluster_index:
            self.logger.warning("没有聚类数据，回退到原始方法")
            return self._find_similarity_groups(similarity_threshold, min_size)

        # 使用新的高效算法
        similar_groups = SimilarityMatcher.find_similar_images_in_all_clusters(
            cluster_index=self.asset_manager.cluster_index,
            similarity_threshold=similarity_threshold
        )

        # 转换格式以匹配现有接口
        formatted_groups = []
        for group in similar_groups:
            if len(group) >= min_size:
                formatted_group = []
                for result in group:
                    formatted_group.append({
                        'image_path': result.path,
                        'similarity': result.similarity,
                        'confidence': result.confidence
                    })
                formatted_groups.append(formatted_group)

        self.logger.info(f"优化算法找到 {len(formatted_groups)} 个相似组")
        return formatted_groups

    def _generate_html_report(self, groups: List[List[Dict[str, Any]]], output_path: str, similarity_threshold: float, lightweight: bool = False):
        """生成HTML报告 - 优化字符串拼接性能"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 统计信息
        total_images = sum(len(group) for group in groups)
        max_group_size = max(len(group) for group in groups) if groups else 0

        # 使用列表收集HTML片段，避免大量字符串拼接
        html_parts = []

        # HTML头部
        html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>相似图片分组报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .stats {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .group {{ background: white; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .group-header {{ background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
        .group-title {{ font-size: 16px; font-weight: bold; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }}
        .image-card {{ text-align: center; border: 2px solid #ddd; padding: 10px; border-radius: 8px; background-color: #fafafa; transition: border-color 0.3s ease, transform 0.2s ease; position: relative; }}
        .image-card:hover {{ border-color: #3498db; transform: translateY(-2px); }}
        .image-card img {{ max-width: 100%; height: 150px; object-fit: contain; border-radius: 4px; cursor: pointer; }}
        .image-card img:hover {{ opacity: 0.8; }}
        .no-thumbnail {{ height: 150px; display: flex; align-items: center; justify-content: center; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d; font-size: 18px; cursor: pointer; }}
        .no-thumbnail:hover {{ background-color: #d5dbdb; }}
        .filename {{ font-size: 12px; margin: 8px 0 4px 0; word-break: break-all; color: #2c3e50; font-weight: bold; cursor: pointer; padding: 2px 4px; border-radius: 3px; display: inline-block; }}
        .filename:hover {{ background-color: #3498db; color: white; }}
        .filepath {{ font-size: 10px; margin: 4px 0; word-break: break-all; color: #7f8c8d; font-family: monospace; line-height: 1.2; cursor: pointer; padding: 2px 4px; border-radius: 3px; }}
        .filepath:hover {{ background-color: #27ae60; color: white; }}
        .metrics {{ margin-top: 8px; display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }}
        .similarity-score {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold; }}
        .confidence-score {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold; }}
        .toggle-btn {{ background-color: #3498db; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; }}
        .toggle-btn:hover {{ background-color: #2980b9; }}
        .group-content {{ display: block; transition: all 0.3s ease; }}
        .group-content.hidden {{ display: none; }}
        .group-collapsed {{ padding: 10px 20px; }}
        .tooltip {{ position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%); background-color: #34495e; color: white; padding: 4px 8px; border-radius: 4px; font-size: 10px; white-space: nowrap; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 1000; }}
        .image-card:hover .tooltip {{ opacity: 1; }}
    </style>
    <script>
        function toggleGroup(groupId) {{
            const content = document.getElementById('group-' + groupId);
            const btn = document.getElementById('btn-' + groupId);
            const groupDiv = document.getElementById('groupdiv-' + groupId);

            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                btn.textContent = '收起';
                if (groupDiv) groupDiv.classList.remove('group-collapsed');
            }} else {{
                content.classList.add('hidden');
                btn.textContent = '展开';
                if (groupDiv) groupDiv.classList.add('group-collapsed');
            }}
        }}

        function openImage(imagePath) {{
            // 直接打开图片文件
            if (window.electron) {{
                window.electron.openPath(imagePath);
            }} else {{
                window.open('file:///' + imagePath.replace(/\\\\/g, '/'));
            }}
        }}

        function copyFilename(filename, event) {{
            event.stopPropagation();
            navigator.clipboard.writeText(filename).then(() => {{
                showTooltip('文件名已复制: ' + filename);
            }}).catch(() => {{
                prompt('文件名:', filename);
            }});
        }}

        function openFolder(imagePath, event) {{
            event.stopPropagation();
            // 打开文件所在文件夹
            if (window.electron) {{
                window.electron.showItemInFolder(imagePath);
            }} else {{
                const folderPath = imagePath.substring(0, imagePath.lastIndexOf('\\\\') || imagePath.lastIndexOf('/'));
                navigator.clipboard.writeText(folderPath).then(() => {{
                    showTooltip('文件夹路径已复制: ' + folderPath);
                }}).catch(() => {{
                    prompt('文件夹路径:', folderPath);
                }});
            }}
        }}

        function showTooltip(message) {{
            const tooltip = document.createElement('div');
            tooltip.style.cssText = `
                position: fixed; top: 20px; right: 20px; z-index: 9999;
                background: #2c3e50; color: white; padding: 10px 15px;
                border-radius: 5px; font-size: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            `;
            tooltip.textContent = message;
            document.body.appendChild(tooltip);

            setTimeout(() => {{
                if (document.body.contains(tooltip)) {{
                    document.body.removeChild(tooltip);
                }}
            }}, 3000);
        }}

        // 键盘快捷键
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                // ESC键收起所有展开的组（保持前3个展开）
                document.querySelectorAll('.group-content:not(.hidden)').forEach((content, index) => {{
                    if (index > 2) {{
                        const groupId = content.id.split('-')[1];
                        toggleGroup(groupId);
                    }}
                }});
            }}
        }});
    </script>
</head>
<body>
    <div class="header">
        <h1>🎯 相似图片分组报告</h1>
    </div>

    <div class="stats">
        <h3>📊 分析统计</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div><strong>相似性阈值:</strong> {similarity_threshold:.3f}</div>
            <div><strong>相似组数量:</strong> {len(groups)}</div>
            <div><strong>包含图片总数:</strong> {total_images}</div>
            <div><strong>最大组大小:</strong> {max_group_size}</div>
            <div><strong>平均组大小:</strong> {total_images / len(groups) if groups else 0:.1f}</div>
        </div>
    </div>
""")

        # 分组内容 - 分批处理减少内存压力
        self.logger.info(f"开始生成 {len(groups)} 个相似组的HTML内容")

        for i, group in enumerate(groups):
            if i % 100 == 0:  # 每100个组输出一次进度
                self.logger.info(f"正在处理第 {i+1}/{len(groups)} 组")

            avg_similarity = sum(item['similarity'] for item in group) / len(group)

            # 组头部
            html_parts.append(f"""
    <div class="group" id="groupdiv-{i}">
        <div class="group-header">
            <div class="group-title">📁 组 {i+1} - {len(group)} 张图片 (平均相似度: {avg_similarity:.3f})</div>
            <button class="toggle-btn" id="btn-{i}" onclick="toggleGroup({i})">收起</button>
        </div>
        <div class="group-content" id="group-{i}">
            <div class="image-grid">""")

            # 处理组内图片 - 批量处理
            image_cards = []
            for item in group:
                image_path = item['image_path']
                try:
                    filename = os.path.basename(image_path)
                    display_path = image_path.replace('\\', '/')
                    js_safe_path = image_path.replace("'", "\\'").replace("\\", "\\\\")

                    # 获取置信度信息（如果有的话）
                    confidence = item.get('confidence', 0.95)  # 默认置信度

                    if lightweight:
                        # 轻量级模式：不生成缩略图，使用文件路径
                        image_cards.append(f"""
                <div class="image-card">
                    <div class="no-thumbnail" onclick="openImage('{js_safe_path}')">📷 {filename}</div>
                    <div class="filename" onclick="copyFilename('{filename}', event)" title="点击复制文件名">{filename}</div>
                    <div class="filepath" onclick="openFolder('{js_safe_path}', event)" title="点击打开文件夹">{display_path}</div>
                    <div class="metrics">
                        <span class="similarity-score">相似度: {item['similarity']:.3f}</span>
                        <span class="confidence-score">置信度: {confidence:.3f}</span>
                    </div>
                </div>""")
                    else:
                        # 生成缩略图
                        img = Image.open(image_path)
                        img.thumbnail((200, 200), Image.Resampling.LANCZOS)

                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_data = buffer.getvalue()
                        img_base64 = base64.b64encode(img_data).decode()

                        image_cards.append(f"""
                <div class="image-card">
                    <img src="data:image/png;base64,{img_base64}" alt="{filename}" onclick="openImage('{js_safe_path}')" title="点击打开图片">
                    <div class="filename" onclick="copyFilename('{filename}', event)" title="点击复制文件名">{filename}</div>
                    <div class="filepath" onclick="openFolder('{js_safe_path}', event)" title="点击打开文件夹">{display_path}</div>
                    <div class="metrics">
                        <span class="similarity-score">相似度: {item['similarity']:.3f}</span>
                        <span class="confidence-score">置信度: {confidence:.3f}</span>
                    </div>
                </div>""")
                except Exception as e:
                    self.logger.warning(f"无法处理图片 {image_path}: {e}")
                    continue

            # 一次性添加所有图片卡片
            html_parts.extend(image_cards)

            # 组尾部
            html_parts.append("""
            </div>
        </div>
    </div>""")

        # HTML尾部
        html_parts.append("""
</body>
</html>""")

        # 一次性连接所有HTML片段并写入文件
        self.logger.info(f"正在写入HTML文件: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))

        self.logger.info(f"HTML报告已生成: {output_path}")