from __future__ import print_function
from math import pi
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
import pointnet_feat as pn
import softpool as sp
from other_models.MSN import msn
# import MSN.MDS.MDS_module as MDS_module
import MSN.expansion_penalty.expansion_penalty_module as expasion
# from MSN.MDS.MDS_module import MinimumDensitySampling as mds

import argparse


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Network(nn.Module):
    def __init__(self,
                 npoints=8192,
                 n_regions=8,
                 dim_pn=512,
                 sp_points=1024,
                 model_lists=['softpool', 'msn', 'folding', 'grnet']):
        super(Network, self).__init__()
        self.do_segment = False
        self.npoints_in = 2048
        self.npoints = npoints
        self.dim_pn = dim_pn
        self.n_regions = n_regions
        self.sp_points = sp_points
        self.n_regions = n_regions
        self.model_lists = model_lists

        if ('softpool' in self.model_lists):
            self.sp_enc = sp.Encoder_softpool(
                regions=self.n_regions,
                npoints=self.npoints_in,
                sp_ratio=self.n_regions,
                dim_pn=self.dim_pn)
            self.sp_dec = sp.Decoder_softpool(
                regions=self.n_regions,
                npoints=self.npoints_in,
                sp_ratio=self.n_regions,
                dim_pn=self.dim_pn)
        if ('folding' in self.model_lists):
            self.pn_enc = nn.Sequential(
                pn.PointNetFeat(npoints, 1024), nn.Linear(1024, dim_pn),
                nn.BatchNorm1d(dim_pn), nn.ReLU())
            self.decoder_fold = msn.PointGenCon(
                bottleneck_size=2 + self.dim_pn)
        if ('msn' in self.model_lists):
            self.pn_enc = nn.Sequential(
                pn.PointNetFeat(npoints, 1024), nn.Linear(1024, dim_pn),
                nn.BatchNorm1d(dim_pn), nn.ReLU())
            self.expansion = expasion.expansionPenaltyModule()
            self.msn = msn.MSN()

        if ('grnet' in self.model_lists):
            from GRNet import grnet
            self.grnet = grnet.GRNet()
        if ('im_grnet' in self.model_lists):
            # NOTE: 2D encoder
            from pix2vox.encoder import Encoder
            self.encoder_img = Encoder()

            from GRNet import grnet
            self.grnet = grnet.GRNet()

        if ('pcn' in self.model_lists):
            from pcn import PCN
            self.pcn = PCN().cuda()

        if ('disp3d' in self.model_lists):
            import displace.encode as disp3d
            self.disp_enc = disp3d.Encoder(support_num=10, neighbor_num=20)

            # NOTE: initial setting for disp3d without image inputs
            #     degrees=[1, 2, 2, 2, 2, 4, 64],
            from displace.decode import Decoder
            self.disp_dec = Decoder(
                features=[1024, 256, 256, 256, 128, 128, 128, 3],
                degrees=[1, 2, 2, 2, 2, 4, 64],
                support=10,
                root_num=1)

            if self.do_segment is True:
                self.disp_seg = disp3d.Disp3D(
                    class_num=12, support_num=10, neighbor_num=20)

        if ('im_disp3d' in self.model_lists):
            from pix2vox.encoder import Encoder
            self.encoder_img = Encoder()

            # NOTE: initial setting for disp3d without image inputs
            #     degrees=[1, 2, 2, 2, 2, 4, 64],
            from displace.decode import Decoder
            self.disp_dec = Decoder(
                features=[1024, 256, 256, 256, 128, 128, 128, 3],
                degrees=[1, 2, 2, 2, 2, 4, 64],
                support=10,
                root_num=1)

        if ('shapegf' in self.model_lists):
            import yaml
            import importlib

            with open('other_models/shapegf/shapenet_recon.yaml', 'r') as f:
                cfg = yaml.load(f)
            cfg = dict2namespace(cfg)
            cfg.log_name = "logs/val"
            cfg.save_dir = "logs/val"
            cfg.log_dir = "logs/val"
            trainer_lib = importlib.import_module(cfg.trainer.type)
            self.trainer = trainer_lib.Trainer(cfg)
            self.enc = self.trainer.encoder
            self.dec = self.trainer.score_net
            self.sigma = self.trainer.sigmas

        if ('vrcnet' in self.model_lists):
            import munch
            import yaml
            args = munch.munchify(
                yaml.safe_load(open('other_models/VRCNet/vrcnet.yaml')))
            from VRCNet.vrcnet import Model
            self.vrcnet = Model(args)

            if self.do_segment is True:
                import displace.encode as disp3d
                self.disp_seg = disp3d.Disp3D(
                    class_num=12, support_num=10, neighbor_num=20)

        if ('pointr' in self.model_lists):
            import yaml
            from pointr.PoinTr import PoinTr
            self.pointr = PoinTr(
                dict2namespace(
                    yaml.load(
                        open('other_models/pointr/PoinTr.yaml'))['model']))
            # self.pointr = PoinTr(dict2namespace(yaml.load(open('other_models/pointr/ftrans.yaml'))['model']))

        if ('im_pointr' in self.model_lists):
            # NOTE: 2D encoder: ResNet
            from pix2vox.encoder import Encoder
            self.encoder_img = Encoder()
            """
            # NOTE: 2D encoder: ConvTransformer
            from pix2vox.model import Model_encoder, Bottleneck
            self.encoder_img = Model_encoder(Bottleneck, [3, 4, 6, 3])
            """

            # NOTE: 3D decoder
            import yaml
            from imgpointr.PoinTr import PoinTr
            self.pointr = PoinTr(
                dict2namespace(
                    yaml.load(
                        open('other_models/imgpointr/PoinTr.yaml'))['model']))

        if ('snowflake' in self.model_lists):
            from snowflake.model import SnowflakeNet
            self.snowflake = SnowflakeNet(dim_feat=512, up_factors=[2, 2])

        if ('im_snowflake' in self.model_lists):
            # NOTE: 2D encoder
            from pix2vox.encoder import Encoder
            self.encoder_img = Encoder()
            # self.pooler = nn.MaxPool2d(2, stride=2)

            from snowflake.model import SnowflakeNet
            # NOTE: 2.5D encoder + 3D decoder
            self.snowflake = SnowflakeNet(
                dim_feat=1024, up_factors=[2, 2], global_feat=True)

    def forward(self, part, images=[]):
        output = {
            'softpool': [],
            'msn': [],
            'folding': [],
            'grnet': [],
            'shapegf': [],
            'disp3d': [],
            'pcn': [],
            'pointr': []
        }
        if ('msn' in self.model_lists):
            # transpose part when using displace
            pn_feat = self.pn_enc(part)
            [pcd_msn1, pcd_msn2, loss_mst, mean_mst_dis] = self.msn(
                part, pn_feat)

        if ('softpool' in self.model_lists):
            input_chosen, feat_softpool = self.sp_enc(part=part)
            pcd_softpool, pcd_fusion, loss_mst = self.sp_dec(
                feature=feat_softpool, part=part)

        if ('folding' in self.model_lists):
            # transpose part when using displace
            pn_feat = self.pn_enc(part)
            mesh_grid = torch.meshgrid([
                torch.linspace(0.0, 1.0, 64),
                torch.linspace(0.0, 1.0, self.npoints // 64)
            ])
            mesh_grid = torch.cat(
                (torch.reshape(mesh_grid[0], (self.npoints, 1)),
                 torch.reshape(mesh_grid[1], (self.npoints, 1))),
                dim=1)
            mesh_grid = torch.transpose(mesh_grid, 0, 1).unsqueeze(0).repeat(
                part.shape[0], 1, 1).cuda()
            pn_feat = pn_feat.unsqueeze(2).expand(
                part.size(0), self.dim_pn, self.npoints).contiguous()
            y = torch.cat((mesh_grid, pn_feat), 1).contiguous()
            pcd_fold_t = self.decoder_fold(y)
            pcd_fold = pcd_fold_t.transpose(1, 2).contiguous()

        if ('grnet' in self.model_lists):
            [pcd_grnet_voxel, pcd_grnet_fine, voxels] = self.grnet(
                part.transpose(1, 2))

        if ('im_grnet' in self.model_lists):
            imgs_feat = self.encoder_img(images)
            # torch.Size([batch, 16, 256, 2, 2])

            imgs_feat = torch.max(imgs_feat, dim=1, keepdim=True).values
            imgs_feat = torch.reshape(imgs_feat, (part.shape[0], 1024))
            [pcd_grnet_voxel, pcd_grnet_fine, voxels] = self.grnet(
                partial_cloud=part.transpose(1, 2), global_feature=imgs_feat)

        if ('pointcnn' in self.model_lists):
            pcd_pcnn = self.pointcnn(part.transpose(1, 2))

        if ('pcn' in self.model_lists):
            pcn_coarse, pcn_fine = self.pcn(part)
        if ('disp3d' in self.model_lists):
            disp_feat, pcd_anchors = self.disp_enc(part.transpose(1, 2))
            pcd_disp = self.disp_dec([disp_feat])
            if self.do_segment is True:
                seg_disp = self.disp_seg(pcd_disp)

        if ('im_disp3d' in self.model_lists):
            imgs_feat = self.encoder_img(images)
            # torch.Size([batch, 16, 256, 4, 4])

            imgs_feat = torch.max(imgs_feat, dim=1, keepdim=True).values
            imgs_feat = torch.reshape(imgs_feat, (part.shape[0], 1, 1024))
            pcd_disp = self.disp_dec([imgs_feat])

        if ('vrcnet' in self.model_lists):
            # pcd_vrcnet_coarse, pcd_vrcnet_fine, pcd_vrcnet_3, pcd_vrcnet_4 = self.vrcnet(part)['out1'], self.vrcnet(part)['out2'], self.vrcnet(part)['out3'], self.vrcnet(part)['out4']
            pcd_vrcnet_coarse, pcd_vrcnet_fine, pcd_vrcnet_3, pcd_vrcnet_4 = self.vrcnet(
                part)
            if self.do_segment is True:
                seg_vrcnet = self.disp_seg(pcd_vrcnet)
        if ('pointr' in self.model_lists):
            pcd_pointr = self.pointr(part.transpose(1, 2))
        if ('im_pointr' in self.model_lists):
            imgs_feats = []
            num_frames = 16
            """
            for i in range(num_frames):
                imgs_feat = self.encoder_img(images[:,i,:,:,:].transpose(1, 3))
                imgs_feats.append(imgs_feat)
            imgs_feat = torch.stack(imgs_feats)
                # shape [2, 1024, 4, 4]
            imgs_feat = torch.reshape(imgs_feat, (part.shape[0], 256, num_frames * 4 * 4))
            """
            # NOTE encoding with ResNet
            imgs_feat = self.encoder_img(images)
            # torch.Size([batch, 16, 256, 4, 4])

            imgs_feat = imgs_feat.transpose(1, 2)
            # NOTE reshape for PE feature
            imgs_feat = torch.reshape(imgs_feat,
                                      (part.shape[0], 256, num_frames * 2 * 2))
            # torch.Size([batch, 15*16, 256])

            pcd_pointr = self.pointr(imgs_feat.transpose(1, 2))

        if ('snowflake' in self.model_lists):
            pcd_snowflake = self.snowflake(point_cloud=part.transpose(1, 2))

        if ('im_snowflake' in self.model_lists):
            imgs_feat = self.encoder_img(images)
            # torch.Size([batch, 16, 256, 2, 2])

            imgs_feat = torch.max(imgs_feat, dim=1, keepdim=True).values
            imgs_feat = torch.reshape(imgs_feat, (part.shape[0], 1, 1024))
            # NOTE: depends whether use point cloud and images as input together
            # pcd_snowflake = self.snowflake(global_feature=imgs_feat.transpose(1, 2))
            pcd_snowflake = self.snowflake(
                point_cloud=part.transpose(1, 2),
                global_feature=imgs_feat.transpose(1, 2))

            # pcd_snowflake_3d = self.snowflake(point_cloud=part.transpose(1, 2))

        # start to organize
        output['softpool'] = [
            pcd_softpool, pcd_fusion, input_chosen, loss_mst
        ] if ('softpool' in self.model_lists) else []
        output['msn'] = [pcd_msn1, pcd_msn2, loss_mst
                         ] if ('msn' in self.model_lists) else []
        output['folding'] = [pcd_fold
                             ] if ('folding' in self.model_lists) else []
        output['grnet'] = [pcd_grnet_voxel, pcd_grnet_fine, voxels
                           ] if ('grnet' in self.model_lists) else []
        output['im_grnet'] = [pcd_grnet_voxel, pcd_grnet_fine, voxels
                              ] if ('im_grnet' in self.model_lists) else []
        output['shapegf'] = self.trainer if (
            'shapegf' in self.model_lists) else []
        output['pcn'] = [pcn_coarse, pcn_fine
                         ] if ('pcn' in self.model_lists) else []
        output['disp3d'] = [pcd_disp
                              ] if ('disp3d' in self.model_lists) else []
        output['im_disp3d'] = [pcd_disp] if (
            'im_disp3d' in self.model_lists) else []
        output['vrcnet'] = [
            pcd_vrcnet_coarse, pcd_vrcnet_fine, pcd_vrcnet_3, pcd_vrcnet_4
        ] if ('vrcnet' in self.model_lists) else []
        output['pointr'] = [pcd_pointr[0], pcd_pointr[1], pcd_pointr[2]
                            ] if ('pointr' in self.model_lists) else []
        output['im_pointr'] = [pcd_pointr[0], pcd_pointr[1]
                               ] if ('im_pointr' in self.model_lists) else []
        output['snowflake'] = pcd_snowflake if (
            'snowflake' in self.model_lists) else []
        output['im_snowflake'] = [pcd_snowflake] if (
            'im_snowflake' in self.model_lists) else []
        return output
