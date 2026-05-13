"""
MuSc 特征提取器 - 第一步：特征提取与 LNAMD 多尺度融合
实现 CLIP、DINOv2、DINOv3 的特征提取和多尺度邻域聚合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union
import sys
import os

# 添加 backbone 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backbone'))

# 初始化全局变量，防止导入失败时未定义
_backbones = None
open_clip = None

try:
    import models.backbone._backbones as _backbones
    import models.backbone.open_clip as open_clip
except ImportError:
    # 如果在外部调用，可能需要调整导入路径
    # 变量已初始化为 None，所以可以安全继续
    pass


class FeatureExtractor:
    """模型加载器，支持 CLIP、DINOv2、DINOv3"""

    def __init__(self, model_type: str, model_name: Optional[str] = None,
                 feature_layer: Optional[Union[int, List[int]]] = None, device: Optional[str] = None):
        """
        初始化特征提取器

        Args:
            model_type: 模型类型，支持 "clip", "dinov2", "dinov3"
            model_name: 具体模型名称（可选）
            feature_layer: 要提取的特征层索引，从1开始，-1表示最后一层。
                          可以是单个整数或整数列表。None表示使用默认层（最后一层）。
            device: 设备，如 "cuda:0" 或 "cpu"
        """
        self.model_type = model_type.lower()
        self.model_name = model_name
        # 将 feature_layer 转换为列表形式
        if feature_layer is None:
            self.feature_layer = [-1]  # 默认最后一层
        elif isinstance(feature_layer, int):
            self.feature_layer = [feature_layer]
        else:
            self.feature_layer = list(feature_layer)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        # 根据模型类型加载模型
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> nn.Module:
        """加载指定类型的模型"""
        if self.model_type == "clip":
            return self._load_clip()
        elif self.model_type == "dinov2":
            return self._load_dinov2()
        elif self.model_type == "dinov3":
            return self._load_dinov3()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}，支持的类型: clip, dinov2, dinov3")

    def _load_clip(self) -> nn.Module:
        """加载 CLIP 模型"""
        # 检查 open_clip 模块是否可用
        if open_clip is None:
            raise ImportError("open_clip模块未导入，请检查models/backbone/open_clip目录")

        # 参考 musc.py 中的实现，使用 open_clip
        if self.model_name is None:
            # 默认使用 CLIP ViT-L/14
            self.model_name = "ViT-L-14"

        # 图像尺寸需要根据模型确定
        image_size = 336  # CLIP ViT-L/14 的默认尺寸

        try:
            # 使用 open_clip 加载
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                image_size,
                pretrained='openai'  # 默认使用 openai 预训练权重
            )
            self.preprocess = preprocess
            model = model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"加载 CLIP 模型失败: {e}")

    def _load_dinov2(self) -> nn.Module:
        """加载 DINOv2 模型"""
        # 使用本地 _backbones.py 中的加载逻辑
        if self.model_name is None:
            # 默认使用 DINOv2 ViT-L/14
            self.model_name = "dinov2_vitl14"

        try:
            model = _backbones.load(self.model_name)
            model = model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"加载 DINOv2 模型失败: {e}")

    def _load_dinov3(self) -> nn.Module:
        """加载 DINOv3 模型"""
        # 使用 timm 或 transformers 加载（官方推荐）
        if self.model_name is None:
            # 默认使用 DINOv3 ViT-L/14
            self.model_name = "vit_large_patch14_dinov3"

        try:
            # 尝试使用 timm 加载
            import timm
            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            model = model.to(self.device)
            return model
        except ImportError:
            try:
                # 如果 timm 不可用，尝试使用 transformers
                from transformers import Dinov3Model
                model = Dinov3Model.from_pretrained("facebook/dinov3-large")
                model = model.to(self.device)
                return model
            except ImportError as e:
                raise ImportError("加载 DINOv3 需要安装 timm 或 transformers 库")

    @torch.no_grad()
    def extract_patch_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多层 patch tokens（不含 [CLS] token）

        Args:
            images: 输入图像张量，形状 [B, 3, H, W]

        Returns:
            patch_tokens_list: 多层 patch tokens 列表，每个元素形状 [B, N, D]
        """
        images = images.to(self.device)

        if self.model_type == "clip":
            return self._extract_clip_tokens(images)
        elif self.model_type == "dinov2":
            return self._extract_dinov2_tokens(images)
        elif self.model_type == "dinov3":
            return self._extract_dinov3_tokens(images)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _extract_clip_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        """提取 CLIP 的多层 patch tokens"""
        # 确定要提取的层索引列表（从1开始，-1表示最后一层）
        layer_indices = []
        for layer in self.feature_layer:
            if layer == -1:
                # 获取视觉编码器的总层数
                if hasattr(self.model.visual, 'depth'):
                    depth = self.model.visual.depth
                elif hasattr(self.model.visual, 'trunk') and hasattr(self.model.visual.trunk, 'depth'):
                    depth = self.model.visual.trunk.depth
                elif hasattr(self.model.visual, 'blocks'):
                    depth = len(self.model.visual.blocks)
                else:
                    # 默认假设 24 层（CLIP ViT-L/14）
                    depth = 24
                layer_index = depth  # 最后一层索引等于总层数
            else:
                layer_index = layer + 1  #  # 从1开始的索引
            layer_indices.append(layer_index)

        # encode_image 返回 (image_features, patch_tokens_list)
        # patch_tokens_list 是列表，每个元素对应一个层的特征
        try:
            # 尝试调用 encode_image，可能返回两个值或一个值

            output = self.model.encode_image(images, layer_indices)
            # output1 = self.model.visual(images, layer_indices)
            if isinstance(output, tuple) and len(output) == 2:
                _, patch_tokens_list = output
            else:
                # 如果只返回一个值，假设是 patch_tokens_list 或单个特征
                if isinstance(output, list):
                    patch_tokens_list = output
                else:
                    # 单个张量，可能是最后一层的特征
                    patch_tokens_list = [output]

            # 确保 patch_tokens_list 是列表
            if not isinstance(patch_tokens_list, list):
                patch_tokens_list = [patch_tokens_list]

            # 去除每个层的 CLS token（第一个 token）
            result = []
            for patch_tokens in patch_tokens_list:
                if patch_tokens.dim() == 3 and patch_tokens.shape[1] > 1:
                    # 形状 [B, N+1, D]，去除 CLS token
                    patch_tokens = patch_tokens[:, 1:, :]
                result.append(patch_tokens)
            return result
        except Exception as e:
            # 如果 encode_image 失败，尝试备用方法
            # 使用视觉 trunk 提取特征，逐层提取
            result = []
            for layer_index in layer_indices:
                try:
                    features = self.model.visual(images, layer_idx=layer_index)
                except TypeError:
                    # 可能不支持 layer_idx 参数，只返回最后一层
                    features = self.model.visual(images)
                # 假设 features 的形状是 [B, N+1, D]
                if features.dim() == 3 and features.shape[1] > 1:
                    features = features[:, 1:, :]
                result.append(features)
            return result

    def _extract_dinov2_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        """提取 DINOv2 的多层 patch tokens"""
        # 确定要提取的层索引列表
        layer_indices = []
        for layer in self.feature_layer:
            if layer == -1:
                # 提取最后一层
                layer_index = self.model.n_blocks  # 总块数
            else:
                layer_index = layer
            layer_indices.append(layer_index)

        # 注意：get_intermediate_layers 的 n 参数是从后往前的索引
        # 如果指定多个层，需要转换为从后往前的索引列表
        # 但 get_intermediate_layers 可以接受 n 为列表，返回多个层
        n_list = layer_indices

        try:
            # 使用 get_intermediate_layers 提取多个层
            # n 可以是列表，返回多个层的特征
            patch_tokens_list = self.model.get_intermediate_layers(
                images,
                n=n_list,
                return_class_token=False
            )

            # 如果返回的是元组，可能是 (tokens, attn) 格式
            if isinstance(patch_tokens_list, tuple):
                patch_tokens_list = patch_tokens_list[0]

            # 确保返回的是列表
            if not isinstance(patch_tokens_list, list):
                patch_tokens_list = [patch_tokens_list]

            return patch_tokens_list
        except AttributeError:
            # 如果模型没有 get_intermediate_layers 方法，使用 forward_features
            # 只能提取最后一层，所以返回多个相同的层（警告）
            features = self.model.forward_features(images)
            if isinstance(features, dict) and "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
                # 注意：这里返回的是最后一层的特征，重复多次
                return [patch_tokens] * len(self.feature_layer)
            else:
                raise RuntimeError("无法从 DINOv2 模型中提取 patch tokens")

    def _extract_dinov3_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        """提取 DINOv3 的多层 patch tokens"""
        # DINOv3 的提取方式取决于使用的后端（timm 或 transformers）
        # 目前 DINOv3 模型通常只提供最后一层特征
        # 为了支持多层，我们返回相同的特征重复多次
        try:
            # 首先尝试 timm 模型
            features = self.model(images)

            # timm 模型设置 num_classes=0 时通常返回 [B, N+1, D]
            if len(features.shape) == 3:
                # 有序列维度，假设第一个是 CLS token
                if features.shape[1] > 1:
                    features = features[:, 1:, :]
                # 返回多个相同的特征
                return [features] * len(self.feature_layer)
            else:
                # 如果没有序列维度，可能需要通过其他方式获取 patch tokens
                # 尝试获取 patch_embed 和 pos_embed 信息
                raise RuntimeError("DINOv3 模型未返回序列特征")
        except Exception:
            try:
                # 尝试 transformers 模型
                outputs = self.model(images)
                # transformers 的 Dinov3Model 返回 BaseModelOutputWithPooling
                last_hidden_state = outputs.last_hidden_state
                if last_hidden_state.shape[1] > 1:
                    last_hidden_state = last_hidden_state[:, 1:, :]
                # 返回多个相同的特征
                return [last_hidden_state] * len(self.feature_layer)
            except Exception as e:
                raise RuntimeError(f"提取 DINOv3 patch tokens 失败: {e}")


class LNAMD(nn.Module):
    """局部邻域多尺度聚合模块"""

    def __init__(self, r_list: List[int] = None):
        """
        初始化 LNAMD 模块

        Args:
            r_list: 聚合半径列表，默认 [1, 3, 5]
        """
        super().__init__()
        if r_list is None:
            r_list = [1, 3, 5]
        self.r_list = r_list

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        对 patch tokens 进行多尺度邻域聚合

        Args:
            patch_tokens: 输入 patch tokens，形状 [B, N, D]

        Returns:
            fused_features: 融合特征，形状 [B, N, D * len(r_list)]
        """
        B, N, D = patch_tokens.shape

        # 假设特征图是方形的（ViT 通常输出方形特征图）
        h = w = int(math.sqrt(N))
        if h * w != N:
            raise ValueError(f"patch 数量 {N} 不是完全平方数，无法 reshape 为方形特征图")

        # 将特征 reshape 为 2D 特征图 [B, D, h, w]
        features_2d = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)  # [B, D, h, w]

        aggregated_features = []
        for r in self.r_list:
            if r == 1:
                # 恒等映射
                agg = features_2d
            else:
                # 平均池化，保持空间尺寸不变
                padding = r // 2
                agg = F.avg_pool2d(features_2d, kernel_size=r, stride=1, padding=padding)
            aggregated_features.append(agg)

        # 在通道维度拼接所有尺度的特征
        fused = torch.cat(aggregated_features, dim=1)  # [B, D * len(r_list), h, w]

        # 转换回 [B, N, D * len(r_list)] 形状
        fused = fused.permute(0, 2, 3, 1).reshape(B, N, len(self.r_list), D)

        return fused.detach()


class MuScFeatureExtractor(nn.Module):
    """MuSc 特征提取器（特征提取 + LNAMD 融合）"""

    def __init__(self, model_type: str, model_name: Optional[str] = None,
                 feature_layer: Union[int, List[int]] = -1, r_list: List[int] = None,
                 device: Optional[str] = None):
        """
        初始化 MuSc 特征提取器

        Args:
            model_type: 模型类型，"clip", "dinov2", "dinov3"
            model_name: 具体模型名称（可选）
            feature_layer: 要提取的特征层索引，从1开始，-1表示最后一层。
                           可以是单个整数或整数列表。
            r_list: LNAMD 聚合半径列表，默认 [1, 3, 5]
            device: 设备，如 "cuda:0" 或 "cpu"
        """
        super().__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.feature_layer = feature_layer
        self.r_list = r_list if r_list is not None else [1, 3, 5]

        # 初始化特征提取器和 LNAMD 模块
        self.feature_extractor = FeatureExtractor(
            model_type, model_name, feature_layer, device
        )
        self.lnamd = LNAMD(r_list=self.r_list)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播：提取多层特征并进行 LNAMD 融合

        Args:
            images: 输入图像，形状 [B, 3, H, W]

        Returns:
            fused_features: 融合后的特征，形状 [B, N, D_fused]
                           D_fused = D * len(r_list) * L，其中 L 是层数
        """
        with torch.no_grad():
            # 提取多层 patch tokens（列表，每个元素形状 [B, N, D]）
            patch_tokens_list = self.feature_extractor.extract_patch_tokens(images)

            # 对每个层的特征分别进行 LNAMD 多尺度融合
            fused_features_list = []
            for patch_tokens in patch_tokens_list:
                fused_features = self.lnamd(patch_tokens)  # [B, N, D*len(r_list)]
                fused_features /= fused_features.norm(dim=-1, keepdim=True)
                fused_features_list.append(fused_features)

            # 在通道维度拼接所有层的融合特征
            fused_features = torch.cat(fused_features_list, dim=2).mean(dim=2)  # [B, N, D*len(r_list)]
            fused_features /= fused_features.norm(dim=-1, keepdim=True)
        return fused_features.detach()      

    def to(self, device):
        """将模型移动到指定设备"""
        super().to(device)
        # 更新特征提取器的设备
        self.feature_extractor.device = torch.device(device)
        self.feature_extractor.model = self.feature_extractor.model.to(device)
        return self


def test_feature_extractor():
    """测试特征提取器（模拟测试，不实际加载模型）"""
    print("=== MuSc 特征提取器测试 ===")
    print("注意：此测试仅验证代码逻辑，需要安装相应库才能实际运行")

    # 测试配置
    test_configs = [
        {"model_type": "clip", "model_name": "ViT-L-14", "feature_layer": [5, 11, 17, 23], "r_list": [1, 3, 5]},
        {"model_type": "dinov2", "model_name": "dinov2_vitl14", "feature_layer": [5, 11, 17, 23], "r_list": [1, 3, 5]},
        # {"model_type": "dinov3", "model_name": "vit_large_patch14_dinov3", "feature_layer":[5, 11, 17, 23], "r_list": [1, 3, 5]},
    ]

    for config in test_configs:
        print(f"\n测试配置: {config}")

        try:
            # 创建特征提取器（不实际加载模型）
            extractor = MuScFeatureExtractor(
                model_type=config["model_type"],
                model_name=config["model_name"],
                feature_layer=config["feature_layer"],
                r_list=config["r_list"],
                device="cpu"
            )

            print(f"  [OK] 特征提取器创建成功")
            print(f"     模型类型: {extractor.model_type}")
            print(f"     模型名称: {extractor.model_name}")
            print(f"     特征层: {extractor.feature_layer}")
            print(f"     LNAMD r_list: {extractor.r_list}")

        except Exception as e:
            print(f"  [FAIL] 特征提取器创建失败: {e}")
            # 对于 DINOv3，可能缺少依赖库
            if config["model_type"] == "dinov3":
                print("     注意: DINOv3 需要 timm 或 transformers 库")


def simple_shape_test():
    """简单的形状测试（使用随机数据）"""
    print("\n=== 形状测试 ===")

    # 模拟输入
    B, C, H, W = 2, 3, 224, 224
    dummy_images = torch.randn(B, C, H, W)

    # 测试 LNAMD 模块
    print("测试 LNAMD 模块...")
    lnamd = LNAMD(r_list=[1, 3, 5])

    # 模拟 patch tokens（假设有 196 个 patches，特征维度 1024）
    N = 196  # 224/16 * 224/16 = 14*14 = 196
    D = 1024
    dummy_patch_tokens = torch.randn(B, N, D)

    with torch.no_grad():
        fused_features = lnamd(dummy_patch_tokens)

    print(f"  输入形状: {dummy_patch_tokens.shape}")
    print(f"  输出形状: {fused_features.shape}")
    print(f"  期望输出形状: [B, N, D*len(r_list)] = [{B}, {N}, {D*3}]")

    assert fused_features.shape == (B, N, D * 3), "LNAMD 输出形状不正确"
    print("  [OK] LNAMD 形状测试通过")

    # 测试多层特征融合
    print("\n测试多层特征融合...")
    L = 3  # 层数
    # 模拟多层特征列表
    dummy_patch_tokens_list = [torch.randn(B, N, D) for _ in range(L)]

    fused_features_list = []
    for patch_tokens in dummy_patch_tokens_list:
        fused_features = lnamd(patch_tokens)
        fused_features_list.append(fused_features)

    # 拼接所有层的融合特征
    fused_all = torch.cat(fused_features_list, dim=-1)
    expected_dim = D * 3 * L
    print(f"  层数: {L}")
    print(f"  每层输入形状: [B, N, D] = [{B}, {N}, {D}]")
    print(f"  每层输出形状: [B, N, D*len(r_list)] = [{B}, {N}, {D*3}]")
    print(f"  最终输出形状: {fused_all.shape}")
    print(f"  期望输出形状: [B, N, D*len(r_list)*L] = [{B}, {N}, {expected_dim}]")

    assert fused_all.shape == (B, N, expected_dim), "多层特征融合输出形状不正确"
    print("  [OK] 多层特征融合形状测试通过")

    # 测试完整的 MuScFeatureExtractor（需要实际模型，这里只演示）
    print("\n测试完整流程（需要安装模型库）...")
    print("  要实际测试，请安装相应库并运行:")
    print("  python models/musc_feature_extractor.py --test")


if __name__ == "__main__":
    # 运行测试
    pass
    # test_feature_extractor()
    # simple_shape_test()