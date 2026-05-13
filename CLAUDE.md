# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MuSc（Multi-Scale）工业异常检测系统。输入图像 → ViT backbone 提取 patch tokens → LNAMD 多尺度邻域聚合 → MSM 互打分 → 输出像素级异常分数与热力图。

## 核心文件与架构

```
musc_feature_extractor.py    # 第一步：FeatureExtractor + LNAMD + MuScFeatureExtractor（三层封装）
template_extractor.py         # 从 train/good 图像批量提取特征，保存为 .pt 模板文件
test_musc_feature_extractor.py # CLIP/DINOv2 特征提取单元测试（用 MVTec 真实图像）
test_bottle_detection.py     # 端到端测试：特征提取 + MSM + 热力图可视化（MVTec bottle）
models/
  musc.py                     # 完整 MuSc 流程：加载数据 → 特征提取 → LNAMD → MSM → RsCIN → 评估/可视化
  backbone/
    _backbones.py             # 统一 backbone 加载器（DINO/DINOv2/timm/CLIP）
    open_clip/                # 本地 open_clip 实现
    dinov2/                   # DINOv2 组件（attention, block, etc.）
  modules/
    _LNAMD.py                 # 完整版 LNAMD（含 PatchMaker+Preprocessing+MeanMapper，支持 r>1 时 patchify）
    _MSM.py                   # MSM 互打分模块（cos/L2，interval averaging，ONNX 兼容）
    _RsCIN.py                 # RsCIN 重打分（基于 CLS token 相似度传播）
    _queue.py                 # FixedSizeFIFOQueue（参考特征队列）
```

## 关键实现差异

**两套 LNAMD 实现：**
- `musc_feature_extractor.py` 中的 `LNAMD`：简化版，用 `avg_pool2d` 实现邻域聚合，假设严格方形特征图。用于独立使用。
- `models/modules/_LNAMD.py` 中的 `LNAMD`：完整版，用 `PatchMaker`（Unfold）+ `Preprocessing`（MeanMapper），支持不规则尺寸插值。用于 `models/musc.py` 的完整流程。

**预处理规范：**
- CLIP：使用 CLIP 内置 `preprocess` 或自定义 `(0.48145466, 0.8278225, 0.9108821) / (0.26862954, 0.26130258, 0.27577711)` 归一化，图像尺寸 336
- DINOv2：使用 ImageNet `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)` 归一化，图像尺寸 224

## 运行方式

```bash
# 端到端测试（需要 MVTec 数据集和 GPU）
python test_bottle_detection.py

# 模板特征提取
python template_extractor.py --model_type clip --template_dir <path/to/train/good> --save_path <output.pt>

# 完整 MuSc 训练+评估（需要 yaml 配置）
python models/musc.py --config <config.yaml>

# 单元测试（仅验证形状逻辑，不加载真实模型）
python -c "from musc_feature_extractor import simple_shape_test; simple_shape_test()"
```

## 依赖

`torch`, `torchvision`, `timm`, `opencv-python`, `Pillow`, `numpy`, `matplotlib`, `openpyxl`, `tqdm`, `transformers`（仅 DINOv3 可选）、`pretrainedmodels`（部分 backbone 可选）

## 约束

- DINO/DINOv2 backbone 从 `torch.hub` 下载预训练权重到本地缓存
- 默认假设方形特征图（`N = h*w` 为完全平方数）
- `musc_feature_extractor.py` 的导入路径依赖项目根目录在 `sys.path` 中
- 目前无正式训练流程——只做特征提取 + 推理/评估
