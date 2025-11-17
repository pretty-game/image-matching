#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片匹配系统 - 简化统一入口
提供最简洁的API接口
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.apis import ImageMatchingAPI
from src.main import GameAssetManager, setup_logging

# 配置日志 - 使用main.py中的统一日志配置
log_file_path = setup_logging()
logger = logging.getLogger(__name__)


def main():
    """主函数 - 简化的命令行接口"""
    parser = argparse.ArgumentParser(
        description='图片匹配系统 - 简化接口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 更新图片库
  python app.py update

  # 生成相似图片报告
  python app.py report

  # 查询相似图片
  python app.py query --image path/to/image.png --top-k 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 更新命令
    update_parser = subparsers.add_parser('update', help='更新图片库')
    update_parser.add_argument('--config', default='./config/settings.yaml', help='配置文件路径')

    # 报告命令
    report_parser = subparsers.add_parser('report', help='生成相似图片报告')
    report_parser.add_argument('--output', default='reports', help='输出目录（默认：reports）')
    report_parser.add_argument('--threshold', type=float, help='相似性阈值')
    report_parser.add_argument('--min-size', type=int, default=2, help='最小组大小')
    report_parser.add_argument('--max-groups', type=int, help='最大组数限制（性能优化）')
    report_parser.add_argument('--lightweight', action='store_true', help='轻量级模式（不生成缩略图）')
    report_parser.add_argument('--no-optimize', action='store_true', help='禁用优化算法（使用原始遍历方法）')
    report_parser.add_argument('--format-type', default='html', help='报告格式（html/json）')
    report_parser.add_argument('--config', default='./config/settings.yaml', help='配置文件路径')

    # 查询命令
    query_parser = subparsers.add_parser('query', help='查找相似图片')
    query_parser.add_argument('--image', required=True, help='查询图片路径')
    query_parser.add_argument('--top-k', type=int, default=10, help='返回结果数量')
    query_parser.add_argument('--threshold', type=float, help='相似性阈值')
    query_parser.add_argument('--config', default='./config/settings.yaml', help='配置文件路径')

    # 状态命令
    status_parser = subparsers.add_parser('status', help='检查系统状态')
    status_parser.add_argument('--config', default='./config/settings.yaml', help='配置文件路径')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logger.info(f"执行命令: {args.command}")

    # 初始化API
    api = ImageMatchingAPI(args.config)

    if args.command == 'update':
        run_update(api)
    elif args.command == 'report':
        run_report(api, args)
    elif args.command == 'query':
        run_query(api, args)


def run_update(api: ImageMatchingAPI):
    """运行更新命令"""
    logger.info("🔧 初始化图片匹配系统...")
    if not api.initialize():
        logger.error("❌ 系统初始化失败")
        return
    logger.info("✅ 系统初始化完成")

def run_report(api: ImageMatchingAPI, args):
    """运行报告命令"""
    logger.info("🔧 初始化图片匹配系统...")
    if not api.initialize():
        logger.error("❌ 系统初始化失败")
        return

    mode_info = []
    if args.max_groups:
        mode_info.append(f"限制{args.max_groups}组")
    if args.lightweight:
        mode_info.append("轻量级模式")
    if args.no_optimize:
        mode_info.append("原始算法")
    else:
        mode_info.append("优化算法")

    mode_str = f" ({', '.join(mode_info)})" if mode_info else ""
    similarity_threshold = 0.9 if args.threshold is None else args.threshold
    logger.info(f"📊 生成相似图片报告 - 阈值: {similarity_threshold}, 模式: {mode_str}")

    result = api.generate_similarity_report(
        output_path=args.output,
        similarity_threshold=similarity_threshold,
        min_group_size=args.min_size,
        max_groups=args.max_groups,
        lightweight=args.lightweight,
        use_optimized=not args.no_optimize,
        format_type=getattr(args, 'format_type', 'html')
    )

    logger.info("="*50)
    logger.info("报告生成结果:")
    logger.info(f"状态: {'✅ 成功' if result['success'] else '❌ 失败'}")

    if result['success']:
        logger.info(f"相似组数量: {result['total_groups']}")
        logger.info(f"相似图片数: {result['total_similar_images']}")
        logger.info(f"相似度比例: {result['similarity_ratio']:.2%}")
        logger.info(f"输出文件: {result['output_path']}")
        logger.info(f"执行时间: {result['execution_time']:.2f}秒")

        if args.lightweight:
            logger.info("💡 轻量级模式已启用，报告中不包含图片缩略图")
        if args.max_groups:
            logger.info(f"⚡ 组数限制已启用，最多显示{args.max_groups}组")
    else:
        error_msg = result.get('error', '未知错误')
        logger.error(f"错误: {error_msg}")


def run_query(api: ImageMatchingAPI, args):
    """运行查询命令"""
    logger.info("🔧 初始化图片匹配系统...")
    if not api.initialize():
        logger.error("❌ 系统初始化失败")
        return

    logger.info(f"🔍 查询相似图片: {args.image}, top_k: {args.top_k}, threshold: {args.threshold}")
    result = api.find_similar_images(
        query_image_path=args.image,
        top_k=args.top_k,
        similarity_threshold=args.threshold
    )

    logger.info("="*50)
    logger.info("查询结果:")
    logger.info(f"状态: {'✅ 成功' if result['success'] else '❌ 失败'}")

    if result['success']:
        similar_images = result['similar_images']
        logger.info(f"找到 {len(similar_images)} 张相似图片, 耗时: {result['execution_time']:.2f}秒")
        logger.info("-" * 80)
        logger.info(f"{'排名':<4} {'文件名':<30} {'相似度':<8} {'文件大小(MB)':<12}")
        logger.info("-" * 80)

        for img in similar_images:
            logger.info(f"{img['rank']:<4} {img['filename'][:29]:<30} "
                       f"{img['similarity']:<8.3f} {img['file_size_mb']:<12.2f}")

        logger.info("\n详细路径:")
        for img in similar_images:
            logger.info(f"{img['rank']}. {img['image_path']}")

        logger.info(f"执行时间: {result['execution_time']:.2f}秒")
    else:
        error_msg = result.get('error', '未知错误')
        logger.error(f"查询失败: {error_msg}")

if __name__ == "__main__":
    main()