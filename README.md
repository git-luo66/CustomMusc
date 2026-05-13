# MuSc — Multi-Scale Industrial Anomaly Detection

基于 ViT 的多尺度工业异常检测系统。输入图像 → ViT backbone 提取 patch tokens → LNAMD 多尺度邻域聚合 → MSM 互打分 → 像素级异常分数与热力图。

## 整体架构

```
图像 [B,3,H,W]
    │
    ▼
┌─────────────────────────┐
│ FeatureExtractor        │  CLIP / DINOv2 / DINOv3
│ 提取多层 patch tokens    │  支持指定中间层
└──────────┬──────────────┘
           │ List([B,N,D]) × L 层
           ▼
┌─────────────────────────┐
│ LNAMD                   │  多尺度邻域平均池化
│ r_list=[1,3,5]          │  通道维拼接 → [B,N,D*3]
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Cross-Layer Fusion      │  跨层均值归约 → [B,N,D]
│ L2 Normalize            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ MSM 互打分              │  Z (query) vs R (reference)
│ cos / L2 distance       │  interval averaging
└──────────┬──────────────┘
           │
           ▼
     异常热力图 [H,W]
```

## 快速开始

### 环境

```bash
pip install torch torchvision timm opencv-python Pillow numpy matplotlib openpyxl tqdm
# 可选: transformers (DINOv3), onnx onnxruntime (ONNX 导出)
```

### 特征提取

```python
from musc_feature_extractor import MuScFeatureExtractor

extractor = MuScFeatureExtractor(
    model_type='clip',           # 'clip' / 'dinov2' / 'dinov3'
    model_name='ViT-L-14',       # backbone 名称
    feature_layer=[5,11,17,23],  # 提取的中间层索引
    r_list=[1,3,5],              # LNAMD 聚合半径
)
features = extractor(images)     # [B, 3, 336, 336] → [B, 576, 1024]
```

### 导出 ONNX

```bash
python export_onnx.py
# 生成:
#   onnx_models/musc_feature_extractor.onnx  # 特征提取
#   onnx_models/msm.onnx                      # 互打分
```

### ONNX 推理测试

```bash
python test_onnx_pipeline.py
# 读取 MVTec bottle 数据，ONNX 推理，输出热力图到 results/bottle_onnx_test/
```

## 核心模块

| 文件 | 说明 |
|------|------|
| `musc_feature_extractor.py` | FeatureExtractor + LNAMD + MuScFeatureExtractor |
| `models/musc.py` | 完整 MuSc 流程（含 RsCIN 重打分、评估指标） |
| `models/modules/_MSM.py` | MSM 互打分（ONNX 兼容） |
| `models/modules/_LNAMD.py` | 完整版 LNAMD（PatchMaker + Preprocessing） |
| `models/modules/_RsCIN.py` | RsCIN 重打分（CLS token 相似度传播） |
| `models/backbone/_backbones.py` | 统一 backbone 加载器 |

## 测试脚本

| 脚本 | 用途 |
|------|------|
| `test_musc_feature_extractor.py` | 特征提取单元测试（MVTec 真实图像） |
| `test_bottle_detection.py` | 端到端异常检测（PyTorch，含热力图） |
| `test_onnx_pipeline.py` | 端到端 ONNX 推理测试 |
| `template_extractor.py` | 从 train/good 提取并保存模板特征 |

## 预处理规范

| Backbone | 图像尺寸 | 归一化 |
|----------|---------|--------|
| CLIP | 336×336 | `(0.482, 0.828, 0.911) / (0.269, 0.261, 0.276)` |
| DINOv2 | 224×224 | `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)` |

## 依赖

`torch` · `torchvision` · `timm` · `opencv-python` · `Pillow` · `numpy` · `matplotlib` · `openpyxl` · `tqdm` · `onnx` (可选) · `onnxruntime` (可选) · `transformers` (可选, DINOv3)

## 约束

- 假设方形特征图（patch 数为完全平方数）
- 目前无训练流程，仅做特征提取 + 推理评估
- DINO/DINOv2 backbone 通过 `torch.hub` 下载预训练权重
