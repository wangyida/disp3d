import torch

def label_points(npoints, divisions):
    labels_for_points = torch.Tensor(
        range(1, (divisions + 1) * (npoints // divisions) + 1))
    labels_for_points = labels_for_points.view(npoints // divisions,
                                               (divisions + 1)).transpose(
                                                   0, 1)
    labels_for_points = (labels_for_points) % (divisions + 1)
    labels_for_points = labels_for_points.contiguous().view(-1)
    return labels_for_points
