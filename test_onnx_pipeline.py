"""
端到端 ONNX 推理测试 — MVTec bottle
使用导出的 ONNX 模型：musc_feature_extractor.onnx + msm.onnx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ============================================================
# 配置
# ============================================================
MODEL_TYPE   = 'clip'
MODEL_NAME   = 'ViT-L-14'
FEATURE_LAYER_LEN = 4
R_LIST_LEN   = 3
TARGET_SIZE  = (336, 336)
PATCH_SIZE   = 14
H_W = TARGET_SIZE[0] // PATCH_SIZE  # 24, spatial grid size
N_PATCHES    = H_W * H_W  # 576

# CLIP 归一化参数
CLIP_MEAN = (0.48145466, 0.8278225, 0.9108821)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# 路径
ONNX_DIR      = r'G:\AIModel-python\CustomMusc\onnx_models'
TEMPLATE_PATH = r'G:\AIModel-python\CustomMusc\templates\bottle_template.pt'
TEST_DIR      = r'D:\PublicDatasets\mvtec_anomaly_detection\bottle\test'
OUTPUT_DIR    = r'G:\AIModel-python\CustomMusc\results\bottle_onnx_test'

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_PER_CAT   = 8   # 每个类别处理的图像数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载 ONNX 模型
# ============================================================
print("=" * 60)
print("加载 ONNX 模型")
print("=" * 60)

# 根据设备选择 ONNX Runtime provider
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
print(f"  Providers: {providers}")

feat_onnx_path = os.path.join(ONNX_DIR, 'musc_feature_extractor.onnx')
msm_onnx_path  = os.path.join(ONNX_DIR, 'msm.onnx')

print(f"  Feature extractor: {feat_onnx_path}")
print(f"  MSM:                {msm_onnx_path}")

sess_feat = ort.InferenceSession(feat_onnx_path, providers=providers)
sess_msm  = ort.InferenceSession(msm_onnx_path, providers=providers)

# 打印模型输入/输出信息
print("\nFeature Extractor ONNX:")
for inp in sess_feat.get_inputs():
    print(f"  Input:  {inp.name}  shape={inp.shape}")
for out in sess_feat.get_outputs():
    print(f"  Output: {out.name}  shape={out.shape}")

print("\nMSM ONNX:")
for inp in sess_msm.get_inputs():
    print(f"  Input:  {inp.name}  shape={inp.shape}")
for out in sess_msm.get_outputs():
    print(f"  Output: {out.name}  shape={out.shape}")

# ============================================================
# 2. 加载模板特征（由 PyTorch 预提取）
# ============================================================
print(f"\n加载模板: {TEMPLATE_PATH}")
template = torch.load(TEMPLATE_PATH, map_location='cpu', weights_only=False)
R = template['features']  # [num_templates, N, D_fused]
print(f"  模板特征: {R.shape}")

# 转 float32 numpy（ONNX Runtime 要求）
R_np = R.float().numpy()
print(f"  R_np: {R_np.shape}  dtype={R_np.dtype}")

# ============================================================
# 3. 图像预处理
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])

# ============================================================
# 4. 主循环 — 对每个类别、每张图像做推理
# ============================================================
categories = sorted([
    d for d in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, d))
])

category_max_scores = {}
total_time_feat = 0
total_time_msm  = 0
total_count     = 0

for category in categories:
    cat_dir = os.path.join(TEST_DIR, category)
    cat_out = os.path.join(OUTPUT_DIR, category)
    os.makedirs(cat_out, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(cat_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])[:NUM_PER_CAT]

    print(f'\n{"="*50}')
    print(f'类别: {category} ({len(image_files)} 张)')
    print(f'{"="*50}')

    max_scores = []

    for img_file in image_files:
        img_path = os.path.join(cat_dir, img_file)

        # ---- 预处理 ----
        image = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 336, 336]

        # ---- 特征提取 (ONNX) ----
        img_np = img_tensor.numpy().astype(np.float32)
        t0 = time.time()
        Z_np = sess_feat.run(None, {'images': img_np})[0]  # [1, N, D_fused]
        t_feat = time.time() - t0
        total_time_feat += t_feat

        # ---- MSM 互打分 (ONNX) ----
        Z_msm = Z_np.astype(np.float32)
        t0 = time.time()
        scores_np = sess_msm.run(None, {
            'Z': Z_msm,
            'R': R_np,
        })[0]  # [1, N]
        t_msm = time.time() - t0
        total_time_msm += t_msm

        total_count += 1

        # ---- 生成异常图 ----
        anomaly_map = scores_np[0].reshape(H_W, H_W)

        max_val = float(anomaly_map.max())
        mean_val = float(anomaly_map.mean())
        max_scores.append(max_val)
        print(f'  {img_file:>20s}  feat={t_feat*1000:5.1f}ms  msm={t_msm*1000:5.1f}ms  '
              f'max={max_val:.6f}  mean={mean_val:.6f}')

        # ---- 上采样 → 热力图 ----
        anomaly_resized = cv2.resize(
            anomaly_map, TARGET_SIZE, interpolation=cv2.INTER_LINEAR
        )

        orig = cv2.imread(img_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, TARGET_SIZE)

        # ---- 可视化：原图 | 异常图 | 叠加 ----
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

        fig.suptitle(f'{category} / {img_file} (ONNX)', fontsize=13, y=0.98)
        plt.tight_layout()

        name_stem = os.path.splitext(img_file)[0]
        plt.savefig(os.path.join(cat_out, f'{name_stem}.jpg'), dpi=120,
                    bbox_inches='tight', pad_inches=0.2)
        plt.close()

    avg = np.mean(max_scores) if max_scores else 0
    std = np.std(max_scores) if max_scores else 0
    category_max_scores[category] = (avg, std)
    print(f'\n  >>> [{category}] 平均 max: {avg:.6f} ± {std:.6f}')

# ============================================================
# 5. 汇总
# ============================================================
print(f'\n{"="*60}')
print('汇总 — 各类别平均最大异常分数 (ONNX)')
print(f'{"="*60}')
for cat, (avg, std) in sorted(category_max_scores.items()):
    label = 'NORMAL ' if cat == 'good' else 'ANOMALY'
    print(f'  [{label}] {cat:<20s}: {avg:.6f} ± {std:.6f}')

print(f'\n平均推理时间 ({total_count} 张):')
print(f'  特征提取: {total_time_feat/total_count*1000:.1f} ms/图')
print(f'  MSM 互打分: {total_time_msm/total_count*1000:.1f} ms/图')
print(f'  总耗时:    {(total_time_feat+total_time_msm)/total_count*1000:.1f} ms/图')

print(f'\n结果保存至: {OUTPUT_DIR}')
