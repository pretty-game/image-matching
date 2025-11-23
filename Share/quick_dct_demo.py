"""
单图DCT可视化快速脚本
用法: python quick_dct_demo.py <图片路径> <dct_size>
示例: python quick_dct_demo.py ./test.png 32
"""
import sys
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def calculate_dct_2d(image_channel):
    """2D DCT变换"""
    return dct(dct(image_channel, axis=0, norm='ortho'), axis=1, norm='ortho')

def calculate_idct_2d(dct_coefficients):
    """2D逆DCT变换"""
    return idct(idct(dct_coefficients, axis=0, norm='ortho'), axis=1, norm='ortho')

def quick_demo(image_path, dct_size):
    """快速演示DCT截断与重建"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 错误: 无法读取图像 {image_path}")
        return

    # 转Lab色彩空间
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(image_lab)

    # 对L通道进行DCT
    L_float = L.astype(np.float32)
    dct_result = calculate_dct_2d(L_float)

    # 截断DCT
    dct_truncated = np.zeros_like(dct_result)
    dct_truncated[:dct_size, :dct_size] = dct_result[:dct_size, :dct_size]

    # 重建L通道
    L_reconstructed = calculate_idct_2d(dct_truncated)
    L_reconstructed = np.clip(L_reconstructed, 0, 255).astype(np.uint8)

    # 对a, b通道同样处理
    reconstructed_channels = [L_reconstructed]
    for channel in [a, b]:
        ch_float = channel.astype(np.float32)
        ch_dct = calculate_dct_2d(ch_float)
        ch_dct_trunc = np.zeros_like(ch_dct)
        ch_dct_trunc[:dct_size, :dct_size] = ch_dct[:dct_size, :dct_size]
        ch_recon = calculate_idct_2d(ch_dct_trunc)
        ch_recon = np.clip(ch_recon, 0, 255).astype(np.uint8)
        reconstructed_channels.append(ch_recon)

    # 合并并转回BGR
    reconstructed_lab = cv2.merge(reconstructed_channels)
    reconstructed_bgr = cv2.cvtColor(reconstructed_lab, cv2.COLOR_Lab2BGR)

    # 可视化DCT系数
    height, width = dct_result.shape
    dct_abs = np.abs(dct_result)
    dct_log = np.log10(dct_abs + 1)
    dct_normalized = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min())

    vis_image = np.zeros((height, width, 3))
    heatmap = plt.cm.jet(dct_normalized[:dct_size, :dct_size])[:, :, :3]
    vis_image[:dct_size, :dct_size] = heatmap
    vis_image[dct_size:, :] = 0.2
    vis_image[:, dct_size:] = 0.2

    # 创建三联图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'原始图像\n({width}×{height})', fontsize=12, weight='bold', fontproperties='SimHei')
    axes[0].axis('off')

    # DCT可视化
    axes[1].imshow(vis_image)
    axes[1].axhline(y=dct_size-0.5, color='white', linewidth=2, linestyle='--')
    axes[1].axvline(x=dct_size-0.5, color='white', linewidth=2, linestyle='--')
    info_retention = (dct_size * dct_size) / (height * width) * 100
    axes[1].set_title(f'DCT截断 ({dct_size}×{dct_size})\n信息保留: {info_retention:.1f}%',
                      fontsize=12, weight='bold', fontproperties='SimHei')
    axes[1].axis('off')

    # 重建图
    axes[2].imshow(cv2.cvtColor(reconstructed_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'重建图像\n({width}×{height})', fontsize=12, weight='bold', fontproperties='SimHei')
    axes[2].axis('off')

    plt.tight_layout()

    # 保存
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_dct{dct_size}_demo.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ 已生成: {output_path}")
    print(f"  DCT尺寸: {dct_size}×{dct_size}")
    print(f"  信息保留率: {info_retention:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python quick_dct_demo.py <图片路径> <dct_size>")
        print("示例: python quick_dct_demo.py ./test.png 32")
        sys.exit(1)

    image_path = sys.argv[1]
    dct_size = int(sys.argv[2])

    quick_demo(image_path, dct_size)
