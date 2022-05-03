import os
import numpy as np
import cv2
import open3d as o3d
import h5py
import torch

def read_images(model_id, num_frames=16):
    file_views = os.listdir(
        os.path.join("/media/wangyida/HDD/database/ShapeNet_RGBs/",
                     '%s' % model_id, "rendering"))
    total_views = len(file_views)
    rendering_image_indexes = range(total_views)
    rendering_images = []
    cnt_images = 0
    for image_idx in rendering_image_indexes:
        if file_views[image_idx][-3:] == 'png':
            # 5 images per object
            if cnt_images == num_frames:
                break
            else:
                cnt_images += 1
            rendering_image = cv2.imread(
                os.path.join(
                    "/media/wangyida/HDD/database/ShapeNet_RGBs/",
                    '%s' % model_id, "rendering",
                    file_views[image_idx]),
                cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                logging.error(
                    'It seems that there is something wrong with the image file %s'
                    % (image_path))
                sys.exit(2)
            rendering_images.append(rendering_image)
    rendering_images = np.asarray(rendering_images)
    return rendering_images

def read_points(filename, dataset):
    if dataset == 'suncg' or dataset == 'fusion' or dataset == '3rscan' or dataset == 'eye':
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        pcd = o3d.io.read_point_cloud(filename)
        coord = torch.from_numpy(np.array(pcd.points)).float()
        color = torch.from_numpy(np.array(pcd.colors)).float()
        return coord, color
    elif dataset == 'shapenet':
        hash_tab = get_hashtab_shapenet()
        fh5 = h5py.File(filename, 'r')
        label = float(hash_tab[filename.split("/")[-2]]['label'])
        coord = torch.from_numpy(np.array(fh5['data'])).float()
        color = torch.from_numpy(
            np.ones_like(np.array(fh5['data'])) / 11 * label).float()
        return coord, color


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]], idx[:n]


def get_hashtab_shapenet():
    hash_tab = {
        'all': {
            'name': 'Test',
            'label': 100,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '04530566': {
            'name': 'Watercraft',
            'label': 1,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '02933112': {
            'name': 'Cabinet',
            'label': 2,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '04379243': {
            'name': 'Table',
            'label': 3,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '02691156': {
            'name': 'Airplane',
            'label': 4,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '02958343': {
            'name': 'Car',
            'label': 5,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '03001627': {
            'name': 'Chair',
            'label': 6,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '04256520': {
            'name': 'Couch',
            'label': 7,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        },
        '03636649': {
            'name': 'Lamp',
            'label': 8,
            'emd1': 0.0,
            'emd2': 0.0,
            'emd3': 0.0,
            'chamfer_dist': 0.0,
            'cnt': 0
        }
    }
    return hash_tab

