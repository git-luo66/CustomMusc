#!/usr/bin/env python3
"""
模板特征提取器
从 MVTEC 数据集的 train/good 目录加载图像作为模板，
使用 MuScFeatureExtractor 提取特征并保存为模板文件。
"""
import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from glob import glob
from typing import Optional, Tuple

from musc_feature_extractor import MuScFeatureExtractor


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    加载单张图像并预处理为模型输入张量

    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (H, W)

    Returns:
        image_tensor: 形状 [1, 3, H, W] 的张量
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    # ImageNet 归一化（CLIP / DINOv2 默认预处理）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def load_template_images(template_dir: str, target_size: Tuple[int, int] = (224, 224),
                         max_images: Optional[int] = None) -> torch.Tensor:
    """
    加载目录中所有模板图像，堆叠为批次张量

    Args:
        template_dir: 模板图像目录路径
        target_size: 目标尺寸 (H, W)
        max_images: 最大加载图像数（None 表示全部加载）

    Returns:
        template_batch: 形状 [N, 3, H, W] 的张量
    """
    image_paths = sorted(glob(os.path.join(template_dir, "*.png")) +
                         glob(os.path.join(template_dir, "*.jpg")) +
                         glob(os.path.join(template_dir, "*.JPEG")))

    if not image_paths:
        raise FileNotFoundError(f"在 {template_dir} 中未找到任何图像")

    if max_images is not None and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]

    print(f"加载 {len(image_paths)} 张模板图像...")

    batch_list = []
    for path in image_paths:
        tensor = load_and_preprocess_image(path, target_size)
        batch_list.append(tensor)

    return torch.cat(batch_list, dim=0)  # [N, 3, H, W]


@torch.no_grad()
def extract_template_features(extractor: MuScFeatureExtractor,
                                template_batch: torch.Tensor,
                                batch_size: int = 8) -> torch.Tensor:
    """
    分批提取模板图像的特征

    Args:
        extractor: MuScFeatureExtractor 实例
        template_batch: 模板图像批次 [N, 3, H, W]
        batch_size: 每批处理图像数

    Returns:
        template_features: 形状 [N, num_patches, D_fused] 的张量
    """
    N = template_batch.shape[0]
    all_features = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = template_batch[start:end].to(extractor.feature_extractor.device)
        features = extractor(batch)  # [B, num_patches, D_fused]
        all_features.append(features.cpu())
        print(f"  已处理: {end}/{N} 张")

    return torch.cat(all_features, dim=0)  # [N, num_patches, D_fused]


def save_template(features: torch.Tensor, save_path: str, config: dict = None):
    """
    保存模板特征到文件

    Args:
        features: 模板特征张量
        save_path: 保存路径（.pt 文件）
        config: 模型配置信息字典
    """
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    data = {
        'features': features,
        'feature_shape': list(features.shape),
        'config': config or {},
    }
    torch.save(data, save_path)
    print(f"模板已保存至: {save_path}")
    print(f"  特征形状: {list(features.shape)}")


def load_template(load_path: str) -> dict:
    """
    加载模板文件

    Args:
        load_path: 模板文件路径（.pt）

    Returns:
        data: 包含 'features' 和 'config' 的字典
    """
    data = torch.load(load_path, map_location='cpu')
    print(f"模板已加载: {load_path}")
    print(f"  特征形状: {data['feature_shape']}")
    if data.get('config'):
        print(f"  模型配置: {data['config']}")
    return data


def main():
    parser = argparse.ArgumentParser(description='MuSc 模板特征提取与保存')
    parser.add_argument('--template_dir', type=str,
                        default=r'D:\PublicDatasets\mvtec_anomaly_detection\bottle\train\good',
                        help='模板图像目录')
    parser.add_argument('--save_path', type=str,
                        default=r'G:\AIModel-python\CustomMusc\templates\bottle_template.pt',
                        help='模板保存路径')
    parser.add_argument('--model_type', type=str, default='clip',
                        choices=['clip', 'dinov2', 'dinov3'],
                        help='backbone 模型类型')
    parser.add_argument('--model_name', type=str, default=None,
                        help='模型名称（默认: dinov2_vitl14 / ViT-L-14）')
    parser.add_argument('--feature_layer', type=int, nargs='+', default=[5, 11, 17, 23],
                        help='提取的特征层索引')
    parser.add_argument('--r_list', type=int, nargs='+', default=[1, 3, 5],
                        help='LNAMD 聚合半径列表')
    parser.add_argument('--target_size', type=int, nargs=2, default=[336, 336],
                        help='输入图像尺寸 (H W)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='特征提取时的批大小')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最大加载图像数')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备（默认自动选择）')

    args = parser.parse_args()

    print("=" * 60)
    print("MuSc 模板特征提取")
    print("=" * 60)

    # 1. 初始化特征提取器
    print(f"\n[1/4] 初始化 {args.model_type.upper()} 特征提取器...")
    extractor = MuScFeatureExtractor(
        model_type=args.model_type,
        model_name=args.model_name,
        feature_layer=args.feature_layer,
        r_list=args.r_list,
        device=args.device,
    )
    print(f"  设备: {extractor.feature_extractor.device}")
    print(f"  模型: {extractor.feature_extractor.model_name}")
    print(f"  特征层: {extractor.feature_layer}")
    print(f"  LNAMD r_list: {extractor.r_list}")

    # 2. 加载模板图像
    print(f"\n[2/4] 加载模板图像...")
    target_size = tuple(args.target_size)
    template_batch = load_template_images(args.template_dir, target_size, args.max_images)
    print(f"  模板批次形状: {list(template_batch.shape)}")

    # 3. 提取特征
    print(f"\n[3/4] 提取模板特征...")
    template_features = extract_template_features(extractor, template_batch, args.batch_size)
    print(f"  原始特征形状: {list(template_features.shape)}")

    # 4. 保存全部 OK 模板特征（不聚合）
    print(f"\n[4/4] 保存模板...")

    actual_model_name = (
        args.model_name
        or getattr(extractor.feature_extractor, 'model_name', None)
    )
    config = {
        'model_type': args.model_type,
        'model_name': actual_model_name,
        'feature_layer': args.feature_layer,
        'r_list': args.r_list,
        'target_size': args.target_size,
        'num_templates': template_batch.shape[0],
    }
    save_template(template_features, args.save_path, config)
    load_template(args.save_path)
    print("\n完成!")


if __name__ == "__main__":
    main()
