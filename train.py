import open3d as o3d
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
from dataset import *
from model import *
from utils import *
import os
import json
import time, datetime
import visdom
from time import time
from other_tools import normalize
sys.path.append("./distance/emd/")
import emd_module as emd
# sys.path.append("./distance/chamfer/")
# import dist_chamfer as cd
sys.path.append("./distance/chamfer_multidim")
from chamfer3D import dist_chamfer_3D as cd
from dataset import resample_pcd


class ModelOptimizer(nn.Module):
    def __init__(self, model):
        super(ModelOptimizer, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()
        self.CD = cd.chamfer_3DDist()  # cd.chamferDist()

    def forward(self, part, gt, part_seg, gt_seg, images):
        eps = 0.005
        iters = 50
        # if part.shape[1] != 3:
        #    part = part.transpose(1, 2)
        images = images[:, :, 5:5 + 128, 5:5 + 128, :]
        output = self.model(part=part, images=images)
        loss_points = torch.zeros(1).cuda()
        loss_others = torch.zeros(1).cuda()
        if output['softpool']:
            dist, _ = self.EMD(output['softpool'][0], gt, eps, iters)
            cdist = torch.sqrt(dist).mean(1)
            if self.model.n_regions == 1:
                dist1, dist2, _, _ = self.CD(
                    output['softpool'][0][:, output['softpool'][0].shape[1] //
                                          2:, :], part)
                cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['softpool'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist
            loss_others += 0.1 * output['softpool'][3]

        if output['msn']:
            dist1, dist2, _, _ = self.CD(output['msn'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['msn'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist
            loss_others += 0.1 * output['msn'][2]

        if output['folding']:
            dist1, dist2, _, _ = self.CD(output['folding'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist

        if output['grnet']:
            from GRNet.extensions.gridding_loss import GriddingLoss
            gridding_loss = GriddingLoss(scales=[64, 128], alphas=[0.5, 0.5])
            dist1, dist2, _, _ = self.CD(output['grnet'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)

            loss_others += gridding_loss(output['grnet'][0], gt)

            dist1, dist2, idx1, _ = self.CD(output['grnet'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            """
            SM = torch.nn.Softmax(dim=-1)
            sem_feat = SM(output['grnet'][2][:, :, :]).float()
            labels_gt = torch.gather(gt_seg[:, :, 0], dim=1, index=idx1.long())
            sem_gt = torch.nn.functional.one_hot(
                labels_gt.to(torch.int64), 12).float()
            loss_sem_fine = torch.mean(-torch.sum(
                0.97 * sem_gt * torch.log(1e-6 + sem_feat) +
                (1 - 0.97) * (1 - sem_gt) * torch.log(1e-6 + 1 - sem_feat),
                dim=-1))
            cdist += 0.01 * loss_sem_fine
            """

            cdist = cdist.mean(0)
            loss_points += cdist

        if output['im_grnet']:
            from GRNet.extensions.gridding_loss import GriddingLoss
            gridding_loss = GriddingLoss(scales=[64], alphas=[0.5])
            dist1, dist2, _, _ = self.CD(output['im_grnet'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)

            loss_others += gridding_loss(output['im_grnet'][0], gt)

            dist1, dist2, idx1, _ = self.CD(output['im_grnet'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)

            cdist = cdist.mean(0)
            loss_points += cdist

        if output['shapegf']:
            updated_gf = output['shapegf'].update(part, gt)
            loss_gf, pcd_shapegf = updated_gf['loss'], updated_gf['x']
            """
            dist, _ = self.EMD(pcd_shapegf, gt, eps, iters)
            cdist = torch.sqrt(dist).mean(1)
            """
            dist1, dist2, _, _ = self.CD(pcd_shapegf, gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            """
            loss_points += cdist
            """
            loss_others += loss_gf

        if output['pcn']:
            dist1, dist2, _, _ = self.CD(output['pcn'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['pcn'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist

        if output['disp3d']:
            """
            dist1, dist2, _, _ = self.CD(output['disp3d'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            # NOTE: original 3D input
            """
            dist, _ = self.EMD(output['disp3d'][0], gt, eps, iters)
            cdist = torch.sqrt(dist).mean(1)
            enforce_sequence = False
            if enforce_sequence is True:
                dist, _ = self.EMD(
                    output['disp3d'][0][:, :output['disp3d'][0].shape[1] //
                                          2, :], part, eps, iters)
                cdist += torch.sqrt(dist).mean(1)
            """
            SM = torch.nn.Softmax(dim=-1)
            sem_feat = SM(output['disp3d'][1][:, :, :]).float()
            _, _, idx1, _ = self.CD(output['disp3d'][0], gt)
            labels_gt = torch.gather(gt_seg[:, :, 0], dim=1, index=idx1.long())
            sem_gt = torch.nn.functional.one_hot(
                labels_gt.to(torch.int64), 12).float()
            loss_sem_fine = torch.mean(-torch.sum(
                0.97 * sem_gt * torch.log(1e-6 + sem_feat) +
                (1 - 0.97) * (1 - sem_gt) * torch.log(1e-6 + 1 - sem_feat),
                dim=-1))
            cdist += 0.01 * loss_sem_fine
            """

            cdist = cdist.mean(0)
            loss_points += cdist

        if output['im_disp3d']:
            """
            dist1, dist2, _, _ = self.CD(output['im_disp3d'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            # NOTE: original 3D input
            """
            dist, _ = self.EMD(output['im_disp3d'][0], gt, eps, iters)
            cdist = torch.sqrt(dist).mean(1)

            cdist = cdist.mean(0)
            loss_points += cdist

        if output['vrcnet']:
            out_vrc, loss_points, loss_others = self.model.vrcnet.trainer(
                part, gt, alpha=0.5)
            loss_points = loss_points[0]
            cdist = loss_points

        if output['pointr']:
            dist1, dist2, _, _ = self.CD(output['pointr'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['pointr'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['pointr'][2], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist

        if output['im_pointr']:
            """
            dist, _ = self.EMD(output['im_pointr'][0], resample_pcd(gt.transpose(0, 1), 1024)[0].transpose(0, 1), eps, iters)
            cdist = torch.sqrt(dist).mean(1)
            """
            dist1, dist2, _, _ = self.CD(output['im_pointr'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_pointr'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist
            """
            for stage in range(len(output['im_pointr'])):
                pts_coord = output['im_pointr'][stage][0].data.cpu(
                )[:, 0:3]
                pts_color = colormap.colormap(
                    output['im_pointr'][stage][0],
                    dataset='shapenet')
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/im_pointr',
                    child='010',
                    pfile='temp' + '-' + str(stage))
            """

        if output['snowflake']:
            dist1, dist2, _, _ = self.CD(output['snowflake'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['snowflake'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['snowflake'][2], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['snowflake'][3], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist
        """
        if output['im_snowflake']:
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][2], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][3], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            cdist = cdist.mean(0)
            loss_points += cdist
        """

        if output['im_snowflake']:
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][0][0], gt)
            cdist = torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][0][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][0][2], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][0][3], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            """
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][1][0], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][1][1], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][1][2], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            dist1, dist2, _, _ = self.CD(output['im_snowflake'][1][3], gt)
            cdist += torch.mean(dist1, 1) + torch.mean(dist2, 1)
            """
            cdist = cdist.mean(0)
            loss_points += cdist

        return cdist, loss_points, loss_others


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument(
        '--workers',
        type=int,
        help='number of data loading workers',
        default=12)
    parser.add_argument(
        '--nepoch',
        type=int,
        default=750,
        help='number of epochs to train for')
    parser.add_argument(
        '--model', type=str, default='', help='optional reload model path')
    parser.add_argument(
        '--npoints',
        nargs='+',
        default=['2048', '4096'],
        help='a pair of numbers for in/out points')
    parser.add_argument(
        '--n_regions', type=int, default=16, help='number of surface elements')
    parser.add_argument(
        '--dataset',
        type=str,
        default="shapenet",
        help='dataset for evaluation')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['softpool', 'msn', 'folding', 'grnet'],
        help='a list of methods')
    parser.add_argument(
        '--savepath', type=str, default='', help='path for saving')

    opt = parser.parse_args()
    print(opt)

    # vis = visdom.Visdom(port = 8097, env=opt.methods) # set your port
    now = datetime.datetime.now()
    save_path = opt.savepath  # now.isoformat()
    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    dir_name = os.path.join('log', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')
    os.system('cp ./train.py %s' % dir_name)
    os.system('cp ./dataset.py %s' % dir_name)
    os.system('cp ./model.py %s' % dir_name)

    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    best_val_loss = 10

    dataset = ShapeNet(
        train=True, npoints=opt.npoints, dataset_name=opt.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    dataset_test = ShapeNet(train=False, npoints=opt.npoints)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

    len_dataset = len(dataset)
    print("Train set size: ", len_dataset)
    network = Network(
        npoints=int(opt.npoints[1]),
        n_regions=opt.n_regions,
        model_lists=opt.methods)
    network = torch.nn.DataParallel(ModelOptimizer(network))
    network.cuda()
    # network.module.model.apply(weights_init)  #initialization of the weight

    if opt.model != '':
        network.module.model.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    lrate = 1e-4
    # 1e-3 for shapeGF
    # lrate = 1e-3
    optimizer = optim.Adam(
        network.module.model.parameters(),
        lr=lrate,
        weight_decay=0,
        betas=(.9, .999))

    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()
    with open(logname, 'a') as f:  #open and append
        f.write(str(network.module.model) + '\n')

    train_curve = []
    val_curve = []
    labels_generated_points = torch.Tensor(
        range(1, (opt.n_regions + 1) * (int(opt.npoints[1]) // opt.n_regions) +
              1)).view(
                  int(opt.npoints[1]) // opt.n_regions,
                  (opt.n_regions + 1)).transpose(0, 1)
    labels_generated_points = (labels_generated_points) % (opt.n_regions + 1)
    labels_generated_points = labels_generated_points.contiguous().view(-1)

    for epoch in range(opt.nepoch):
        #TRAIN MODE
        # train_loss.reset()
        network.module.model.train()

        # learning rate schedule
        if epoch == 20:
            optimizer = optim.Adam(
                network.module.model.parameters(), lr=lrate / 10.0)
        if epoch == 40:
            optimizer = optim.Adam(
                network.module.model.parameters(), lr=lrate / 100.0)

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            id, part, gt, part_seg, gt_seg, images = data
            # id, part, gt, part_seg, gt_seg = data
            part = part.float().cuda()
            part_seg = part_seg.float().cuda()
            gt = gt.float().cuda()
            gt_seg = gt_seg.float().cuda()

            # Rescale and center each point cloud
            sample_mean, sample_scale = normalize.normalize(part)
            if opt.dataset != '3rscan':
                sample_scale = torch.ones_like(sample_scale)
                sample_mean = torch.zeros_like(sample_mean)
            part = (part - sample_mean) / sample_scale
            gt = (gt - sample_mean) / sample_scale

            if opt.methods[0] == 'shapegf':
                part = part.transpose(1, 2)
            cdist, loss_points, loss_others = network(
                part.transpose(1, 2), gt, part_seg, gt_seg, images)

            loss_all = loss_points + loss_others
            # loss_all.backward() # single GPU
            loss_all.sum().backward()
            # train_loss.update(cdist.mean().item())
            optimizer.step()

            if i % 10 == 0:
                idx = random.randint(0, part.size()[0] - 1)
            # print((epoch*len_dataset/opt.batchSize+i) % 300)
            if (epoch * len_dataset + i) % 300 == 0 or i == 0:
                print('saving net...')
                torch.save(network.module.model.state_dict(),
                           '%s/network.pth' % (dir_name))

            """
            print(opt.methods[0] + ' train [%d: %d/%d]  chamfer: %.2f' %
                  (epoch, i, len_dataset / opt.batchSize,
                   loss_points.mean().item() * 1e4))
            """
            print(opt.methods[0] + ' train [%d: %d/%d]  chamfer: %.2f' %
                  (epoch, i, len_dataset / opt.batchSize,
                   cdist.mean().item() * 1e4))
        # train_curve.append(train_loss.avg)

        # VALIDATION
        if epoch % 200 == 199:
            val_loss.reset()
            network.module.model.eval()
            with torch.no_grad():
                for i, data in enumerate(dataloader_test, 0):
                    id, part, gt, part_seg, gt_seg, images = data
                    part = part.float().cuda()
                    part_seg = part_seg.float().cuda()
                    gt = gt.float().cuda()
                    gt_seg = gt_seg.float().cuda()
                    if opt.methods[0] == 'shapegf':
                        part = part.transpose(1, 2)
                    cdist, chamfer_dist, _ = network(
                        part.transpose(2, 1), gt, part_seg, gt_seg, images)
                    idx = random.randint(0, part.size()[0] - 1)
                    print(opt.methods[0] + ' val [%d: %d/%d]  chamfer: %.2f' %
                          (epoch, i, len_dataset / opt.batchSize,
                           chamfer_dist.mean().item()))


if __name__ == "__main__":
    main()
