"""
MuSc 异常检测测试脚本 — MVTec bottle
使用 MuScFeatureExtractor 提取特征 + MSM 互打分 + 热力图可视化
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ====== 配置（与模板一致） ======
MODEL_TYPE = 'clip'
MODEL_NAME = 'ViT-L-14'
FEATURE_LAYER = [5, 11, 17, 23]
R_LIST = [1, 3, 5]
TARGET_SIZE = (336, 336)
PATCH_SIZE = 336 // 14  # 24

# CLIP 归一化参数
CLIP_MEAN = (0.48145466, 0.8278225, 0.9108821)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# 路径
TEMPLATE_PATH = r'G:\AIModel-python\CustomMusc\templates\bottle_template.pt'
TEST_DIR      = r'D:\PublicDatasets\mvtec_anomaly_detection\bottle\test'
OUTPUT_DIR    = r'G:\AIModel-python\CustomMusc\results\bottle_test'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_PER_CATEGORY = 8  # 每个类别处理的图像数

# ====== 加载模块 ======
sys.path.insert(0, r'G:\AIModel-python\CustomMusc')
from musc_feature_extractor import MuScFeatureExtractor
from models.modules._MSM import MSM

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'设备: {DEVICE}')
print('加载 MuSc 特征提取器 ...')
extractor = MuScFeatureExtractor(
    model_type=MODEL_TYPE,
    model_name=MODEL_NAME,
    feature_layer=FEATURE_LAYER,
    r_list=R_LIST,
    device=DEVICE,
)
extractor.eval()

print(f'加载模板: {TEMPLATE_PATH}')
template = torch.load(TEMPLATE_PATH, map_location=DEVICE, weights_only=False)
R = template['features']  # [209, 576, 1024]
print(f'模板特征: {R.shape}')

msm = MSM(topmin_min=0, topmin_max=0.3, choice='cos').eval()

preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# ====== 主循环 ======
categories = sorted([
    d for d in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, d))
])

category_max_scores = {}  # 记录每类的平均最大分数

for category in categories:
    cat_dir = os.path.join(TEST_DIR, category)
    cat_out = os.path.join(OUTPUT_DIR, category)
    os.makedirs(cat_out, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(cat_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])[:NUM_PER_CATEGORY]

    print(f'\n{"="*50}')
    print(f'类别: {category} ({len(image_files)} 张)')
    print(f'{"="*50}')

    max_scores = []

    for img_file in image_files:
        img_path = os.path.join(cat_dir, img_file)

        # 1. 预处理
        image = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # [1,3,336,336]

        # 2. 提取特征
        with torch.no_grad():
            Z = extractor(img_tensor)  # [1, 576, 1024]

        # 3. MSM 互打分（Z 需要扩展到与 R 相同 batch 维度）
        with torch.no_grad():
              # [209, 576, 1024]
            anomaly_scores = msm(Z, R)  # [1, 576]

        # 4. 生成异常图
        anomaly_map = anomaly_scores.reshape(PATCH_SIZE, PATCH_SIZE).cpu().numpy()  # [24, 24]

        max_val = float(anomaly_map.max())
        mean_val = float(anomaly_map.mean())
        max_scores.append(max_val)
        print(f'  {img_file:>20s}  max={max_val:.6f}  mean={mean_val:.6f}')

        # 5. 上采样 → 热力图
        anomaly_resized = cv2.resize(
            anomaly_map, TARGET_SIZE, interpolation=cv2.INTER_LINEAR
        )

        # 读取原图
        orig = cv2.imread(img_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, TARGET_SIZE)

        # 6. 画图：原图 | 异常图 | 叠加
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        vmax = max(anomaly_resized.max(), 1e-6)

        axes[0].imshow(orig)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')

        im1 = axes[1].imshow(anomaly_resized, cmap='jet', vmin=0, vmax=vmax)
        axes[1].set_title(f'Anomaly Map\nmax={max_val:.4f}', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        axes[2].imshow(orig)
        im2 = axes[2].imshow(anomaly_resized, cmap='jet', alpha=0.5, vmin=0, vmax=vmax)
        axes[2].set_title('Overlay', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        fig.suptitle(f'{category} / {img_file}', fontsize=13, y=0.98)
        plt.tight_layout()

        name_stem = os.path.splitext(img_file)[0]
        plt.savefig(os.path.join(cat_out, f'{name_stem}.jpg'), dpi=120,
                    bbox_inches='tight', pad_inches=0.2)
        plt.close()

    avg = np.mean(max_scores) if max_scores else 0
    std = np.std(max_scores) if max_scores else 0
    category_max_scores[category] = (avg, std)
    print(f'\n  >>> 平均 max 分数: {avg:.6f} ± {std:.6f}')

# ====== 汇总 ======
print(f'\n{"="*60}')
print('汇总 — 各类别平均最大异常分数')
print(f'{"="*60}')
for cat, (avg, std) in sorted(category_max_scores.items()):
    label = 'NORMAL' if cat == 'good' else 'ANOMALY'
    print(f'  [{label}] {cat:<20s}: {avg:.6f} ± {std:.6f}')

print(f'\n结果保存至: {OUTPUT_DIR}')
