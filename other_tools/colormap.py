import sys
import numpy as np
import matplotlib
sys.path.append("./distance/chamfer_multidim")
from chamfer3D import dist_chamfer_3D as cd
CD = cd.chamfer_3DDist()

def colormap(points, gt=[], gt_seg=[], with_fp=False, dataset='shapenet'):
    if (gt != [] and gt_seg == []):
            raise AssertionError()
    elif gt != [] and gt_seg != []:
        dist1, dist2, idx1, _ = CD.forward(
            input1=points,
            input2=gt)
        # NOTE: render with category labels
        if dataset == 'shapenet':
            pts_color = matplotlib.cm.tab10(gt_seg[0, idx1.long(), 0].cpu().numpy()[0] / 11)[:, :3]
        elif dataset == 'eye':
            pts_color = matplotlib.cm.rainbow(gt_seg[0, idx1.long(), 0].cpu().numpy()[0] / 11)[:, :3]
        else:
            pts_color = matplotlib.cm.rainbow(gt_seg[0, idx1.long(), 0].cpu().numpy()[0] / 11)[:, :3]
        # NOTE: render false positive points
        if with_fp:
            pts_color[:] = [0.5,0.5,0.5]
            pts_color[dist1[0].cpu().numpy() > 0.001] = [0.5,0,0]
    else:
        negative_shift = -0.5
        if points.data.cpu().numpy().shape[0] == 1:
            points = points.data.cpu().numpy()[0, :, 0:3]
        else:
            points = points.data.cpu().numpy()[:, 0:3]
        pts_color = np.array(points - negative_shift)
        pts_color = np.clip(pts_color, 0.001, 1.0)
        norm = np.sqrt(np.sum(pts_color**2, axis=1, keepdims=True))
        pts_color /= norm
    return pts_color
