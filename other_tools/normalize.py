import torch
def normalize(pcds):
    mean = torch.mean(pcds, dim=1, keepdim=True)
    scale = torch.max(
        torch.norm(
            torch.abs(pcds - mean),
            p=2,
            dim=2,
            keepdim=True),
        dim=1,
        keepdim=True)[0] / 0.5
    # pcds_out = (pcds - mean) / scale
    return mean, scale
