"""
DCT截断与重建可视化脚本
用于演示不同DCT尺寸对图像重建质量的影响
"""
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def calculate_dct_2d(image_channel):
    """对单通道图像进行2D DCT变换"""
    return dct(dct(image_channel, axis=0, norm='ortho'), axis=1, norm='ortho')

def calculate_idct_2d(dct_coefficients):
    """对DCT系数进行2D逆变换"""
    return idct(idct(dct_coefficients, axis=0, norm='ortho'), axis=1, norm='ortho')

def visualize_dct_colorful(dct_result, dct_size, save_path):
    """
    生成彩色可视化的DCT系数图
    左上角有效区域用彩色显示,其余区域用灰色
    """
    height, width = dct_result.shape

    # 创建可视化图像
    fig, ax = plt.subplots(figsize=(8, 8))

    # 归一化DCT系数到0-1范围用于显示
    dct_abs = np.abs(dct_result)
    dct_log = np.log10(dct_abs + 1)  # 对数变换增强可视化
    dct_normalized = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min())

    # 创建RGB图像
    vis_image = np.zeros((height, width, 3))

    # 有效区域(左上角dct_size×dct_size)用彩色热力图
    heatmap = plt.cm.jet(dct_normalized[:dct_size, :dct_size])[:, :, :3]
    vis_image[:dct_size, :dct_size] = heatmap

    # 无效区域用深灰色
    vis_image[dct_size:, :] = 0.2
    vis_image[:, dct_size:] = 0.2

    # 显示
    ax.imshow(vis_image)
    ax.axhline(y=dct_size-0.5, color='white', linewidth=2, linestyle='--')
    ax.axvline(x=dct_size-0.5, color='white', linewidth=2, linestyle='--')

    # 添加文字说明
    ax.text(dct_size/2, -10, f'保留区域: {dct_size}×{dct_size}',
            ha='center', fontsize=12, color='red', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(width-10, height+10, f'舍弃区域',
            ha='right', fontsize=10, color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'DCT频域系数 (保留 {dct_size}×{dct_size}/{height}×{width})',
                 fontsize=14, weight='bold', fontproperties='SimHei')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ DCT可视化已保存: {save_path}")

def process_image_with_dct(image_path, dct_size, output_dir):
    """
    处理图像: 提取DCT系数, 生成截断DCT可视化图, 重建图像

    Args:
        image_path: 输入图像路径
        dct_size: DCT保留尺寸 (如8, 16, 32, 64)
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 读取图像
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    height, width = image.shape[:2]
    print(f"\n处理图像: {image_path.name}")
    print(f"原始尺寸: {width}×{height}, DCT保留尺寸: {dct_size}×{dct_size}")

    # 转换到Lab色彩空间
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 分离通道
    L, a, b = cv2.split(image_lab)

    # 对每个通道进行DCT变换
    channels = [L, a, b]
    channel_names = ['L', 'a', 'b']
    reconstructed_channels = []

    for i, (channel, name) in enumerate(zip(channels, channel_names)):
        # 转换为float32
        channel_float = channel.astype(np.float32)

        # DCT变换
        dct_result = calculate_dct_2d(channel_float)

        # 保存第一个通道(L通道)的DCT可视化
        if i == 0:
            dct_vis_path = output_dir / f"{image_path.stem}_dct_{dct_size}x{dct_size}_visual.png"
            visualize_dct_colorful(dct_result, dct_size, dct_vis_path)

        # 截断: 只保留左上角dct_size×dct_size
        dct_truncated = np.zeros_like(dct_result)
        dct_truncated[:dct_size, :dct_size] = dct_result[:dct_size, :dct_size]

        # 逆DCT重建
        reconstructed = calculate_idct_2d(dct_truncated)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        reconstructed_channels.append(reconstructed)

    # 合并通道
    reconstructed_lab = cv2.merge(reconstructed_channels)

    # 转回BGR
    reconstructed_bgr = cv2.cvtColor(reconstructed_lab, cv2.COLOR_Lab2BGR)

    # 保存重建图像
    reconstructed_path = output_dir / f"{image_path.stem}_reconstructed_{dct_size}x{dct_size}.png"
    cv2.imwrite(str(reconstructed_path), reconstructed_bgr)
    print(f"✓ 重建图像已保存: {reconstructed_path}")

    # 计算信息保留率
    info_retention = (dct_size * dct_size) / (height * width) * 100
    print(f"✓ 信息保留率: {info_retention:.2f}% ({dct_size}×{dct_size}/{height}×{width})")

    # 创建对比图
    create_comparison_image(image, reconstructed_bgr, dct_size, output_dir / f"{image_path.stem}_comparison_{dct_size}x{dct_size}.png")

def create_comparison_image(original, reconstructed, dct_size, save_path):
    """创建原图与重建图的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 原图
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始图像', fontsize=14, weight='bold', fontproperties='SimHei')
    axes[0].axis('off')

    # 重建图
    axes[1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'DCT {dct_size}×{dct_size} 重建', fontsize=14, weight='bold', fontproperties='SimHei')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 对比图已保存: {save_path}")

def batch_process(image_path, dct_sizes, output_dir):
    """
    批量处理多个DCT尺寸

    Args:
        image_path: 输入图像路径
        dct_sizes: DCT尺寸列表, 如[8, 16, 32, 64]
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"批量DCT截断与重建演示")
    print(f"输入图像: {image_path}")
    print(f"DCT尺寸: {dct_sizes}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)

    for dct_size in dct_sizes:
        process_image_with_dct(image_path, dct_size, output_dir)

    print("\n" + "=" * 70)
    print("✓ 全部处理完成!")
    print(f"✓ 输出文件在目录: {output_dir.absolute()}")
    print("=" * 70)

if __name__ == "__main__":
    # 配置参数
    IMAGE_PATH = r"F:\WorkSpace\image_matching_dct\Share\PixPin_2025-11-22_17-25-15_256.png"
    OUTPUT_DIR = r"F:\WorkSpace\image_matching_dct\Share\dct_visualization"
    DCT_SIZES = [8, 16, 32, 64]  # 可以根据需要调整

    # 执行批量处理
    batch_process(IMAGE_PATH, DCT_SIZES, OUTPUT_DIR)
