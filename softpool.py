import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# NOTE: libs for MDS sampling of fine reconstruction with input points
from other_models.MSN import msn
import MSN.expansion_penalty.expansion_penalty_module as expasion
import MSN.MDS.MDS_module as MDS_module


class Sorter_projected(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Sorter_projected, self).__init__()
        self.conv1d = torch.nn.Conv1d(dim_in, dim_out, 1).cuda()

    def forward(self, x):
        val_sort = self.conv1d(x)
        idx_sort = torch.argmax(val_sort, dim=1)
        return val_sort, idx_sort


class Softpool(nn.Module):
    def __init__(self, regions=16, sp_ratio=4, dim_feat=256):
        super(Softpool, self).__init__()
        self.regions = regions
        self.sp_ratio = sp_ratio
        self.dim_feat = dim_feat
        self.sorter = Sorter_projected(self.dim_feat, self.regions)

    def forward(self, x):
        [self.size_bth, _, self.pnt_per_sort] = list(x.shape)
        self.pnt_per_sort //= self.sp_ratio

        val_sort, idx_sort = self.sorter(x)

        # NOTE: initialize softpool feature which is presented as F*
        FEAT_star = torch.zeros(self.size_bth, self.dim_feat, self.regions,
                                self.pnt_per_sort).cuda()
        idx_star = torch.zeros(self.size_bth, self.regions, self.regions,
                               self.pnt_per_sort).cuda()
        for region in range(self.regions):
            val_temp, idx_temp = torch.sort(
                val_sort[:, region, :], dim=1, descending=True)
            idx_filter = idx_temp[:, :self.pnt_per_sort].unsqueeze(1).repeat(
                1, self.dim_feat, 1)

            FEAT_star[:, :, region, :] = torch.gather(
                x, dim=2, index=idx_filter)
            idx_star[:, :, region, :] = idx_temp[:, :self.
                                                 pnt_per_sort].unsqueeze(
                                                     1).repeat(
                                                         1, self.regions, 1)

        return FEAT_star, idx_star, idx_sort


class Encoder_softpool(nn.Module):
    def __init__(self, regions=16, npoints=2048, sp_ratio=8, dim_pn=256):
        super(Encoder_softpool, self).__init__()
        # NOTE parameters for softpool
        # parametric model to produce softpool activations
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, dim_pn, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(dim_pn)

        # hyperparameters for softpool
        self.regions = regions
        self.sp_points = npoints // sp_ratio

        # softpool
        self.softpool = Softpool(
            self.regions, sp_ratio=sp_ratio, dim_feat=dim_pn)

        # NOTE parameters for further encoding
        self.enc_regions = nn.Sequential(
            nn.Conv2d(
                dim_pn + self.regions,
                dim_pn,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='replicate'), nn.LeakyReLU(0.2),
            nn.Conv2d(
                dim_pn,
                512,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='replicate'), nn.LeakyReLU(0.2),
            nn.Conv2d(
                512,
                512,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                padding_mode='replicate'), nn.LeakyReLU(0.2))

        # input for embedding has 32 points now,
        # then in total it is regions x 32 points down-sampled by 2*2*2=8
        ebd_pnt_reg = npoints // (self.regions * 8)
        if self.regions == 1:
            ebd_pnt_out = 256
        elif self.regions > 1:
            ebd_pnt_out = 512

        self.pooling = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(1, ebd_pnt_reg), stride=(1, ebd_pnt_reg)),
            nn.MaxPool2d(
                kernel_size=(1, self.regions), stride=(1, self.regions)),
            nn.ConvTranspose2d(
                512,
                512,
                kernel_size=(1, ebd_pnt_out),
                stride=(1, ebd_pnt_out),
                padding=(0, 0)), nn.LeakyReLU(0.2))

    def forward(self, part):
        # NOTE: produce a code using a MLP
        x = F.relu(self.bn1(self.conv1(part)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # NOTE: sort the features with softpool
        FEAT_star, idx_star, idx_sort = self.softpool(x)

        # NOTE: supplement a softpool onehot code for each point
        seg_sp = torch.nn.functional.one_hot(
            idx_sort.to(torch.int64), self.regions).transpose(1, 2).float()
        seg_sp = seg_sp.unsqueeze(2).repeat(1, 1, self.regions, 1)
        seg_sp = torch.gather(seg_sp, dim=3, index=idx_star.long())
        FEAT_seg_star = torch.cat((FEAT_star, seg_sp), 1).contiguous()

        FEAT_seg_star = FEAT_seg_star.view(FEAT_seg_star.shape[0],
                                           FEAT_seg_star.shape[1], 1,
                                           self.regions * self.sp_points)
        """
        FEAT_star = FEAT_star.view(FEAT_star.shape[0], FEAT_star.shape[1], 1,
                               self.regions * self.sp_points)
        """
        idx_star = idx_star.view(idx_star.shape[0], idx_star.shape[1], 1,
                                 self.regions * self.sp_points)

        # NOTE: choose input points with strong activations
        input_chosen = torch.gather(
            part, dim=2,
            index=idx_star[:, 0:1, 0, :].repeat(1, 3, 1).long()).transpose(
                1, 2)

        # NOTE: further encode the softpool feature
        FEAT_regconv = self.enc_regions(FEAT_seg_star)  # 256 points

        # NOTE: max-pool the latent feature and increase the resolution again
        if self.regions == 1:
            FEAT_out = torch.cat((self.pooling(FEAT_regconv), FEAT_regconv),
                                 dim=-1)  # 512 points
        elif self.regions > 1:
            FEAT_out = self.pooling(FEAT_regconv)  # 512 points

        return input_chosen, FEAT_out


class Decoder_softpool(nn.Module):
    def __init__(self, regions=16, npoints=2048, sp_ratio=8, dim_pn=256):
        super(Decoder_softpool, self).__init__()
        self.npoints = npoints
        self.regions = regions
        self.reg_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                dim_pn, 512, kernel_size=(1, 2), stride=(1, 2), padding=(0,
                                                                         0)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 512, kernel_size=(1, 2), stride=(1, 2), padding=(0, 0)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, dim_pn, kernel_size=(1, 2), stride=(1, 2), padding=(0,
                                                                         0)),
            nn.LeakyReLU(0.2))

        self.expansion = expasion.expansionPenaltyModule()

        self.sp_dec_mlp = msn.PointGenCon(bottleneck_size=dim_pn)
        self.sp_dec_residual = msn.PointNetRes()

    def forward(self, feature, part):
        sp_feat_high = self.reg_deconv(feature)  # 4096 points

        pcd_sp_high_t = self.sp_dec_mlp(sp_feat_high[:, :, 0, :])
        pcd_sp_high = pcd_sp_high_t.transpose(1, 2).contiguous()

        id1 = torch.ones(part.shape[0], 1, part.shape[2]).cuda().contiguous()
        id3 = torch.zeros(pcd_sp_high_t.shape[0], 1,
                          pcd_sp_high_t.shape[2]).cuda().contiguous()
        labeled_observe = torch.cat((part, id1), 1)
        labeled_high = torch.cat((pcd_sp_high_t, id3), 1)
        fusion_high = torch.cat((labeled_observe, labeled_high), 2)

        # Set separation restriction for different regions
        dist, _, mean_mst_dis_h = self.expansion(
            pcd_sp_high, self.npoints // np.max((8, self.regions)), 1.5)
        loss_mst = torch.mean(dist)
        resampled_idx_high = MDS_module.minimum_density_sample(
            fusion_high[:, 0:3, :].transpose(1, 2).contiguous(),
            pcd_sp_high.shape[1], mean_mst_dis_h)
        fusion_high = MDS_module.gather_operation(fusion_high,
                                                  resampled_idx_high)
        pcd_fusion_high = (fusion_high[:, 0:3, :]
                           + self.sp_dec_residual(fusion_high)).transpose(
                               2, 1).contiguous()

        return pcd_sp_high, pcd_fusion_high, loss_mst
