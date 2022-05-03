import numpy as np
import open3d as o3d
import matplotlib.cm


def colormap(xyz):
    negative_shift = -0.5
    vec = np.array(xyz - negative_shift)
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
    vec /= norm
    return vec


"""
kernel = np.load('kernel_grnet.npy')
for i in range(32):
    activation = np.abs(kernel[i][0] - np.mean(kernel[i][0][1:3,1:3,1:3]))
    med = np.median(np.abs(activation))
    activation[activation <= (med+np.max(activation))/2] = 0
    x, y, z = np.where(activation)
    pts_color = matplotlib.cm.autumn((activation[x,y,z] - np.min(activation[x,y,z]))/np.max(activation[x,y,z]))[:,:3]
    x = x-np.random.rand(*x.shape)/4
    y = y-np.random.rand(*y.shape)/4
    z = z-np.random.rand(*z.shape)/4
    x = x-np.random.rand(*x.shape)/4
    y = y-np.random.rand(*y.shape)/4
    z = z-np.random.rand(*z.shape)/4

    coordinates = np.array(([(x/3.0 - 0.5)/3.0, (y/3.0 - 0.5)/3.0, (z/3.0 - 0.5)/3.0]))
    coordinates = np.transpose(coordinates)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.Vector3dVector(coordinates)
    pcd.colors = o3d.Vector3dVector(pts_color)
    o3d.write_point_cloud("./kernel_fusion_grnet%d.pcd" % i, pcd)

"""

receptive = np.load('kernel_shapenet.npy')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.Vector3dVector(np.transpose(receptive))
# pcd.colors = o3d.Vector3dVector(colormap(np.transpose(receptive / 8.0)))
normed = np.sqrt(
    np.sum(np.transpose(receptive / 8.0)**2, axis=1, keepdims=True))
pcd.colors = o3d.Vector3dVector(matplotlib.cm.summer(normed)[:, 0, :3])
o3d.write_point_cloud("./kernel_shapenet.pcd", pcd, write_ascii=True)

directions = np.load('kernel_shapenet_s10.npy')
directions = np.reshape(directions, (3, 10, 32)) * 2
weights = np.load('weight_shapenet_s10.npy')
weights = np.reshape(weights, (10, 32))
weights = weights - np.min(weights, axis=0)
weights /= np.max(weights, axis=0)
pts_color = matplotlib.cm.autumn(weights / 5.0)[:, :, :3]

for i in range(32):
    pcd.points = o3d.Vector3dVector(np.transpose(directions[:, :, i]))
    pcd.colors = o3d.Vector3dVector(pts_color[:, i, :])
    o3d.write_point_cloud(
        "./kernel_shapenet_%d.pcd" % i, pcd, write_ascii=True)

import sys
sys.path.append("./distance/chamfer/")
import dist_chamfer as cd
import torch
CD = cd.chamferDist()
output = np.transpose(directions)
gt = np.repeat(np.expand_dims(np.transpose(receptive), 0), 32, axis=0)
_, dist, idx1, _ = CD.forward(
    input1=torch.tensor(output).cuda(), input2=torch.tensor(gt).cuda())
for i in range(32):
    pcd.points = o3d.Vector3dVector(
        np.append(
            gt[i][idx1[i].cpu()], np.transpose(directions[:, :, i]), axis=0))
    # pcd.colors = o3d.Vector3dVector(np.append(colormap(gt[i][idx1[i].cpu()] / 8.0), pts_color[:,i,:], axis=0))
    normed = np.sqrt(
        np.sum((gt[i][idx1[i].cpu()] / 8.0)**2, axis=1, keepdims=True))
    pcd.colors = o3d.Vector3dVector(
        np.append(
            matplotlib.cm.summer(normed)[:, 0, :3], pts_color[:, i, :],
            axis=0))
    o3d.write_point_cloud(
        "./kernel_shapenet_%d_match.pcd" % i, pcd, write_ascii=True)

# ans = F.normalize(self.gcn_enc.conv_0.directions.cpu().detach(), dim=0).numpy()
