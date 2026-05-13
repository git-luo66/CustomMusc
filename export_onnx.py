"""
Export MuSc pipeline to ONNX:
  1) MuScFeatureExtractor (FeatureExtractor + LNAMD)  -> musc_feature_extractor.onnx
  2) MSM mutual scoring                                -> msm.onnx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from musc_feature_extractor import MuScFeatureExtractor
from models.modules._MSM import MSM

# ============================================================
# Config (consistent with test_bottle_detection.py)
# ============================================================
MODEL_TYPE    = 'clip'
MODEL_NAME    = 'ViT-L-14'
FEATURE_LAYER = [5, 11, 17, 23]
R_LIST        = [1, 3, 5]
TARGET_SIZE   = (336, 336)
PATCH_SIZE    = 14
N_PATCHES     = (TARGET_SIZE[0] // PATCH_SIZE) * (TARGET_SIZE[1] // PATCH_SIZE)  # 576
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onnx_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Export MuScFeatureExtractor -> ONNX
# ============================================================
print("=" * 60)
print("[1/2] Export MuScFeatureExtractor -> ONNX")
print("=" * 60)

print(f"Loading {MODEL_TYPE.upper()} backbone: {MODEL_NAME} ...")
extractor = MuScFeatureExtractor(
    model_type=MODEL_TYPE,
    model_name=MODEL_NAME,
    feature_layer=FEATURE_LAYER,
    r_list=R_LIST,
    device=DEVICE,
)
extractor.eval()


class ExtractorExportWrapper(torch.nn.Module):
    """Clean wrapper without torch.no_grad()/detach() for ONNX tracing."""
    def __init__(self, feat_ext, lnamd_mod):
        super().__init__()
        self.feature_extractor = feat_ext
        self.lnamd = lnamd_mod

    def forward(self, images):
        patch_tokens_list = self.feature_extractor.extract_patch_tokens(images)

        fused_list = []
        for pt in patch_tokens_list:
            f = self.lnamd(pt)
            f = f / f.norm(dim=-1, keepdim=True)
            fused_list.append(f)

        out = torch.cat(fused_list, dim=2).mean(dim=2)
        out = out / out.norm(dim=-1, keepdim=True)
        return out


export_wrapper = ExtractorExportWrapper(extractor.feature_extractor, extractor.lnamd).to(DEVICE)
export_wrapper.eval()

# Freeze all parameters for ONNX export (Dynamo exporter limitation)
print("Freezing all parameters for ONNX export ...")
for p in export_wrapper.feature_extractor.model.parameters():
    p.requires_grad_(False)

# PyTorch forward verification
dummy_img = torch.randn(1, 3, *TARGET_SIZE, device=DEVICE)
print("Testing PyTorch forward ...")
with torch.no_grad():
    pt_out = export_wrapper(dummy_img)
print(f"  Output shape: {list(pt_out.shape)}")
print(f"  Expected:     [1, {N_PATCHES}, D_fused]")

# Export ONNX
feat_onnx = os.path.join(OUTPUT_DIR, 'musc_feature_extractor.onnx')
print(f"Exporting -> {feat_onnx} ...")
torch.onnx.export(
    export_wrapper,
    dummy_img,
    feat_onnx,
    input_names=['images'],
    output_names=['fused_features'],
    dynamic_axes={
        'images': {0: 'batch'},
        'fused_features': {0: 'batch'},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

# Verify
import onnx
onnx_model = onnx.load(feat_onnx)
onnx.checker.check_model(onnx_model)
print("  [OK] ONNX model verified")

# Compare ONNX Runtime vs PyTorch output
import onnxruntime as ort
sess = ort.InferenceSession(feat_onnx, providers=['CPUExecutionProvider'])
dummy_np = dummy_img.cpu().numpy().astype(np.float32)
onnx_out = sess.run(None, {'images': dummy_np})[0]
pt_np = pt_out.cpu().numpy()
max_diff = np.abs(onnx_out - pt_np).max()
status = 'OK' if max_diff < 1e-4 else 'WARN'
print(f"  Max diff (PyTorch vs ONNX): {max_diff:.6e}  [{status}]")

# ============================================================
# 2. Export MSM -> ONNX
# ============================================================
print("\n" + "=" * 60)
print("[2/2] Export MSM -> ONNX")
print("=" * 60)

msm = MSM(topmin_min=0, topmin_max=0.3, choice='cos')
msm.eval()

D_feat = pt_out.shape[-1]
S, B, N = 1, 209, N_PATCHES
dummy_Z = torch.randn(S, N, D_feat)
dummy_R = torch.randn(B, N, D_feat)

print("Testing MSM PyTorch forward ...")
with torch.no_grad():
    msm_out = msm(dummy_Z, dummy_R)
print(f"  Output shape: {list(msm_out.shape)}  (expected [1, {N}])")

msm_onnx = os.path.join(OUTPUT_DIR, 'msm.onnx')
print(f"Exporting -> {msm_onnx} ...")
torch.onnx.export(
    msm,
    (dummy_Z, dummy_R),
    msm_onnx,
    input_names=['Z', 'R'],
    output_names=['anomaly_scores'],
    dynamic_axes={
        'Z': {0: 'S', 1: 'N'},
        'R': {0: 'B', 1: 'N'},
        'anomaly_scores': {1: 'N'},
    },
    opset_version=17,
)

onnx_model = onnx.load(msm_onnx)
onnx.checker.check_model(onnx_model)

sess_msm = ort.InferenceSession(msm_onnx, providers=['CPUExecutionProvider'])
msm_onnx_out = sess_msm.run(None, {
    'Z': dummy_Z.numpy().astype(np.float32),
    'R': dummy_R.numpy().astype(np.float32),
})[0]
max_diff_msm = np.abs(msm_onnx_out - msm_out.numpy()).max()
status = 'OK' if max_diff_msm < 1e-4 else 'WARN'
print(f"  Max diff (PyTorch vs ONNX): {max_diff_msm:.6e}  [{status}]")
print("  [OK] MSM ONNX model verified")

print(f"\n[OK] ONNX models saved to: {OUTPUT_DIR}/")
print(f"     - {feat_onnx}")
print(f"     - {msm_onnx}")
