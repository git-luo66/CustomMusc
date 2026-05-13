import torch


class MSM(torch.nn.Module):
    """Multi-Scale Matching module, ONNX-exportable.

    Computes anomaly scores by comparing query features Z against reference features R
    using either L2 distance or cosine distance with interval averaging.

    Args:
        topmin_min: Lower bound for interval averaging (absolute if >= 1, fraction of B otherwise)
        topmin_max: Upper bound for interval averaging (absolute if >= 1, fraction of B otherwise)
        choice: Distance metric, 'cos' (cosine) or 'L2' (euclidean)

    Input:
        Z: Query features [S, N, D] S表示OK样本数量
        R: Reference features [B, N, D] B表示待检测样本数量

    Output:
        anomaly_scores: [1, N]
    """

    def __init__(self, topmin_min=0, topmin_max=0.3, choice='cos'):
        super().__init__()
        self.topmin_min = topmin_min
        self.topmin_max = topmin_max
        if choice not in ('cos', 'L2'):
            raise ValueError(f"choice must be 'cos' or 'L2', got '{choice}'")
        self.choice = choice

    def forward(self, Z, R):
        B, N, D = R.shape
        Z_expanded = Z.expand(R.shape[0], -1, -1)
        if self.choice == 'cos':
            # Cosine distance: 1 - max cosine similarity per query patch
            # (Z @ R^T) -> [B, N, N]; permute -> [N, B, N]; max over R patches -> [N, B]
            sim = torch.bmm(Z_expanded, R.transpose(1, 2)).permute(1, 0, 2)
            patch2image = 1.0 - sim.max(dim=-1)[0]
        else:
            # L2 distance: min euclidean distance per query patch
            patch2image = torch.cdist(Z_expanded, R).min(dim=-1)[0].transpose(0, 1)

        # Interval average parameters
        k_max = self.topmin_max
        k_min = self.topmin_min
        if k_max < 1:
            k_max = int(B * k_max)
        if k_min < 1:
            k_min = int(B * k_min)
        if k_max < k_min:
            k_max, k_min = k_min, k_max
        # Guard: ensure at least 1 value is selected for the mean
        k_max = max(k_max, k_min + 1)

        # ONNX-compatible: sort + slice instead of torch.topk (avoid dynamic-k issues)
        sorted_vals, _ = torch.sort(patch2image.float(), dim=1)  # [N, B] ascending
        selected = sorted_vals[:, k_min:k_max]  # [N, k_max - k_min]
        scores = selected.mean(dim=1)  # [N]

        return scores.unsqueeze(0)  # [1, N]
