#!/usr/bin/env python3
"""
测试 MuSc 特征提取器：验证 CLIP 和 DINOv2 的特征提取是否正确
使用 MVTEC 数据集中的一张图像进行测试
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from musc_feature_extractor import MuScFeatureExtractor


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    加载图像并预处理为适合模型输入的张量

    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (H, W)

    Returns:
        image_tensor: 形状为 [1, 3, H, W] 的张量
    """
    # 加载图像
    img = Image.open(image_path).convert('RGB')

    # 调整大小
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    # 转换为 numpy 数组并归一化到 [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # 转换为 PyTorch 张量并调整通道顺序 [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # 使用 ImageNet 统计信息进行归一化（CLIP 和 DINOv2 的默认预处理）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def test_model(model_type, model_name, image_tensor, feature_layer=None, r_list=[1, 3, 5]):
    """
    测试特定模型的特征提取

    Args:
        model_type: 模型类型 ('clip' 或 'dinov2')
        model_name: 模型名称
        image_tensor: 输入图像张量
        feature_layer: 要提取的特征层索引
        r_list: LNAMD 半径列表

    Returns:
        features: 提取的特征
        shape_info: 形状信息字典
    """
    print(f"\n=== 测试 {model_type.upper()} ({model_name}) ===")

    # 创建特征提取器
    try:
        extractor = MuScFeatureExtractor(
            model_type=model_type,
            model_name=model_name,
            feature_layer=feature_layer,
            r_list=r_list,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"  设备: {extractor.feature_extractor.device}")
    except Exception as e:
        print(f"  创建特征提取器失败: {e}")
        return None, None

    # 提取特征
    with torch.no_grad():
        try:
            features = extractor(image_tensor)
            print(f"  特征提取成功")
        except Exception as e:
            print(f"  特征提取失败: {e}")
            return None, None

    # 收集形状信息
    B, N, D_fused = features.shape
    H, W = image_tensor.shape[2], image_tensor.shape[3]

    # 根据模型类型推断 patch 大小
    if model_type == 'dinov2':
        # DINOv2 使用 patch 大小 14
        patch_size = 14
    elif model_type == 'clip':
        # CLIP 模型：根据模型名称推断
        if model_name and '14' in model_name:
            patch_size = 14
        elif model_name and '16' in model_name:
            patch_size = 16
        else:
            # 默认使用 14（ViT-L-14）
            patch_size = 14
    else:
        # 其他模型默认 16
        patch_size = 16

    expected_N = (H // patch_size) * (W // patch_size)  # 预期 patch 数量

    shape_info = {
        'batch_size': B,
        'num_patches': N,
        'fused_dim': D_fused,
        'input_height': H,
        'input_width': W,
        'expected_patches': expected_N,
        'patch_size': patch_size,
        'r_list_length': len(r_list)
    }

    print(f"  输入形状: [{B}, 3, {H}, {W}]")
    print(f"  输出形状: [{B}, {N}, {D_fused}]")
    print(f"  预期 patch 数量: {expected_N} (基于 {patch_size}x{patch_size} 的 patch 大小)")
    print(f"  LNAMD 半径列表: {r_list} (融合因子: {len(r_list)})")

    # 验证 patch 数量
    if N == expected_N:
        print(f"  [OK] Patch 数量正确")
    else:
        print(f"  [FAIL] Patch 数量不匹配: 得到 {N}, 预期 {expected_N}")

    # 验证特征维度 (D_fused 应该是原始维度 × len(r_list))
    # 我们不知道原始维度 D，但可以检查 D_fused 是否能被 len(r_list) 整除
    if D_fused % len(r_list) == 0:
        D_original = D_fused // len(r_list)
        print(f"  [OK] 融合维度可被 {len(r_list)} 整除，原始维度大约为 {D_original}")
    else:
        print(f"  [FAIL] 融合维度 {D_fused} 不能被 {len(r_list)} 整除")

    # 计算特征统计信息
    features_np = features.cpu().numpy()
    print(f"  特征统计 - 均值: {features_np.mean():.4f}, 标准差: {features_np.std():.4f}")
    print(f"  特征范围 - 最小值: {features_np.min():.4f}, 最大值: {features_np.max():.4f}")

    return features, shape_info


def main():
    print("=" * 60)
    print("MuSc 特征提取器测试")
    print("=" * 60)

    # 检查依赖项
    print("\n检查依赖项...")
    try:
        import models.backbone._backbones as _backbones
        print("  [OK] _backbones 模块可用")
    except ImportError as e:
        print(f"  [FAIL] _backbones 模块不可用: {e}")
        return

    try:
        import models.backbone.open_clip as open_clip
        print("  [OK] open_clip 模块可用")
    except ImportError as e:
        print(f"  [FAIL] open_clip 模块不可用: {e}")
        return

    # 加载测试图像
    mvtec_path = r"D:\PublicDatasets\mvtec_anomaly_detection"
    test_image_path = os.path.join(mvtec_path, "bottle", "train", "good", "000.png")

    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        # 尝试找其他图像
        import glob
        alt_images = glob.glob(os.path.join(mvtec_path, "*", "train", "good", "*.png"))
        if alt_images:
            test_image_path = alt_images[0]
            print(f"使用替代图像: {test_image_path}")
        else:
            print("未找到任何测试图像")
            return

    print(f"\n使用测试图像: {test_image_path}")
    test_backbone_models = ["clip","dinov2"]
    for test_backbone_model in test_backbone_models:

        # 加载和预处理图像
        try:
            if test_backbone_model == "clip":
            
                image_tensor = load_and_preprocess_image(test_image_path,target_size = (336,336))
                print(f"图像加载成功，形状: {image_tensor.shape}")
            else:
                image_tensor = load_and_preprocess_image(test_image_path)
                print(f"图像加载成功，形状: {image_tensor.shape}")
        except Exception as e:
            print(f"图像加载失败: {e}")
            return

        # 测试 CLIP
        feature_layer = [5, 11, 17, 23]
        if test_backbone_model == "clip":
            clip_features, clip_info = test_model(
                model_type=test_backbone_model,
                model_name='ViT-L-14',
                image_tensor=image_tensor,
                feature_layer=feature_layer,
                r_list=[1, 3, 5]
            )

        # 测试 DINOv2
        else:
            dinov2_features, dinov2_info = test_model(
                model_type=test_backbone_model,
                model_name='dinov2_vitl14',
                image_tensor=image_tensor,
                feature_layer=feature_layer,
                r_list=[1, 3, 5]
            )

if __name__ == "__main__":
    main()