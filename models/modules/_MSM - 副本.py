import torch
from tqdm import tqdm

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""

def compute_scores_fast_L2(Z, R, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    # image_num, patch_num, c = Z.shape
    patch2image = torch.cdist(Z, R)
    patch2image = torch.min(patch2image, -1)[0].permute(1, 0)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)


def compute_scores_fast_cos(Z, R, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, _ = Z.shape
    Z_expanded = Z.expand_as(R)
    tensor = torch.full((patch_num,image_num), 1).to(device)
    patch2image = compute_cosin_score(Z_expanded, R).to(device)
    patch2image = tensor - torch.max(patch2image, -1)[0]
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)



def compute_cosin_score(tensor1, tensor2):
    tensor3 = (tensor1 @ tensor2.permute(0, 2, 1)).permute(1,0,2)
    return tensor3


def MSM(Z, R, device, topmin_min=0, topmin_max=0.3, choice='cos'):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    if choice == 'cos':
        anomaly_scores_i = compute_scores_fast_cos(Z, R, device, topmin_min, topmin_max).unsqueeze(0)
    elif choice == 'L2':
        anomaly_scores_i = compute_scores_fast_L2(Z, R, device, topmin_min, topmin_max).unsqueeze(0)
    anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix



    