import os
import numpy as np
import open3d as o3d

def points_save(points, colors, root='pcds/regions', child='all', pfile=''):
    os.makedirs(root, exist_ok=True)
    os.makedirs(root + '/' + child, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    points_npy = np.float32(points)
    colors_npy = np.float32(colors)
    # ranges_samll = np.where(np.linalg.norm(points_npy, axis=1) < 0.25)
    ranges_samll = np.where(np.linalg.norm(points_npy, axis=1) < 1.25)
    pcd.points = o3d.utility.Vector3dVector(points_npy[ranges_samll])
    pcd.colors = o3d.utility.Vector3dVector(colors_npy[ranges_samll])
    o3d.io.write_point_cloud(
        os.path.join(root, '%s.ply' % pfile),
        pcd,
        write_ascii=True,
        compressed=True)
