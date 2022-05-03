import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import random
from other_tools.dataloader import read_images, read_points, resample_pcd


#from utils import *
class ShapeNet(data.Dataset):
    def __init__(self,
                 train=True,
                 npoints=['2048', '4096'],
                 dataset_name='shapenet'):
        self.dataset = dataset_name
        if train:
            if self.dataset == 'suncg':
                self.list_path = './list_pcd/train_suncg.list'
            elif self.dataset == 'fusion':
                self.list_path = './list_pcd/train_fusion.list'
            elif self.dataset == '3rscan':
                self.list_path = './list_pcd/train_3rscan.list'
            elif self.dataset == 'eye':
                self.list_path = './list_pcd/train_eye.list'
            elif self.dataset == 'shapenet':
                self.list_path = './list_pcd/train_shapenet.list'
        else:
            if self.dataset == 'suncg':
                self.list_path = './list_pcd/valid_suncg.list'
            elif self.dataset == 'fusion':
                self.list_path = './list_pcd/test_fusion.list'
            elif self.dataset == '3rscan':
                self.list_path = './list_pcd/valid_3rscan.list'
            elif self.dataset == 'eye':
                self.list_path = './list_pcd/test_eye.list'
            elif self.dataset == 'shapenet':
                self.list_path = './list_pcd/valid_shapenet.list'
        self.npoints = npoints
        self.train = train

        with open(os.path.join(self.list_path)) as file:
            self.model_list = [line.strip().replace('/', '/') for line in file]
        random.shuffle(self.model_list)
        self.len = len(self.model_list)

    def __getitem__(self, index):
        model_id = self.model_list[index]
        scan_id = index

        if self.train:
            if self.dataset == 'suncg':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_partial/",
                        '%s.pcd' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/train/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == 'fusion':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/train/pcd_partial/",
                        '%s.pcd' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/train/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == '3rscan':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/3RSCAN/train/partial/",
                        '%s.ply' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/3RSCAN/train/complete/",
                        '%s.ply' % model_id), self.dataset)
            elif self.dataset == 'eye':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/eye_fb/train/partial/",
                        '%s.ply' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/eye_fb/train/complete/",
                        '%s.ply' % model_id), self.dataset)
                images = []
            elif self.dataset == 'shapenet':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/train/partial/",
                        '%s.h5' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/train/gt/",
                        '%s.h5' % model_id), self.dataset)
                images = read_images(model_id)[:, :, :, :3]
        else:
            if self.dataset == 'suncg':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial/",
                        '%s.pcd' % model_id))
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == 'fusion':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/test/pcd_partial/",
                        '%s.pcd' % model_id))
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/050_200/test/pcd_complete/",
                        '%s.pcd' % model_id), self.dataset)
            elif self.dataset == '3rscan':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/3RSCAN/test/partial/",
                        '%s.ply' % model_id))
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/3RSCAN/test/complete/",
                        '%s.ply' % model_id), self.dataset)
            elif self.dataset == 'eye':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/eye_fb/test/partial/",
                        '%s.ply' % model_id))
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/eye_fb/test/complete/",
                        '%s.ply' % model_id), self.dataset)
                images = []
            elif self.dataset == 'shapenet':
                part, part_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/partial/",
                        '%s.h5' % model_id), self.dataset)
                comp, comp_color = read_points(
                    os.path.join(
                        "/media/wangyida/HDD/database/shapenet/val/gt/",
                        '%s.h5' % model_id), self.dataset)
                images = read_images(model_id)[:, :, :, :3]

        part_sampled, idx_sampled = resample_pcd(part, int(self.npoints[0]))
        part_seg = np.round(part_color[idx_sampled] * 11)
        comp_sampled, idx_sampled = resample_pcd(comp, int(self.npoints[1]))
        comp_seg = np.round(comp_color[idx_sampled] * 11)
        """
        comp_seg = []
        for i in range (1, 12):
            import ipdb; ipdb.set_trace()
            comp_seg.append(resample_pcd(comp_sampled[comp_color == i], 512))
        """
        # images = images[:, 52: 52+32, 52: 52+32, :]
        return model_id, part_sampled, comp_sampled, part_seg, comp_seg, images

    def __len__(self):
        return self.len
