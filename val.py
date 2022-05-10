import sys
import cv2
import matplotlib.cm
from time import time
from matplotlib import pyplot as plt
import open3d as o3d
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import h5py
import os
import visdom
sys.path.append("./distance/emd/")
import emd_module as emd
# sys.path.append("./distance/chamfer/")
# import dist_chamfer as cd
sys.path.append("./distance/chamfer_multidim")
from chamfer3D import dist_chamfer_3D as cd
from dataset import resample_pcd, read_points
from other_tools.dataloader import get_hashtab_shapenet
from other_tools import colormap, points_save, label_points, normalize
EMD = emd.emdModule()
# CD = cd.chamferDist()
CD = cd.chamfer_3DDist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='./trained_model/network.pth',
        help='optional reload model path')
    parser.add_argument(
        '--npoints',
        nargs='+',
        default=['2048', '4096'],
        help='a pair of numbers for in/out points')
    parser.add_argument(
        '--n_regions',
        type=int,
        default=16,
        help='number of primitives in the atlas')
    parser.add_argument(
        '--env', type=str, default="SoftPool_VAL", help='visdom environment')
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

    opt = parser.parse_args()
    print(opt)

    network = Network(
        npoints=int(opt.npoints[1]),
        n_regions=opt.n_regions,
        model_lists=opt.methods)
    network.cuda()
    # network.apply(weights_init)

    # vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

    if opt.model != '':
        network.load_state_dict(torch.load(opt.model))
        print("Previous weight loaded ")

    if 'im_pointr' not in opt.methods:
        network.eval()
    if opt.dataset == 'suncg':
        with open(os.path.join('./list_pcd/valid_suncg_cvpr.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial/"
        gt_dir = "/media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete/"
    elif opt.dataset == 'fusion':
        with open(os.path.join('./list_pcd/test_fusion_few.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/050_200/test/pcd_partial/"
        gt_dir = "/media/wangyida/HDD/database/050_200/test/pcd_complete/"
    elif opt.dataset == '3rscan':
        with open(os.path.join('./list_pcd/test_3rscan_few.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/3RSCAN/test/partial/"
        gt_dir = "/media/wangyida/HDD/database/3RSCAN/test/complete/"
    elif opt.dataset == 'eye':
        with open(os.path.join('./list_pcd/test_eye.list')) as file:
            model_list = [line.strip().replace('/', '/') for line in file]
        part_dir = "/media/wangyida/HDD/database/eye_fb/test/partial/"
        gt_dir = "/media/wangyida/HDD/database/eye_fb/test/complete/"
    elif opt.dataset == 'shapenet':
        hash_tab = get_hashtab_shapenet()
        complete3d_benchmark = False
        if complete3d_benchmark == True:
            with open(os.path.join('./list_pcd/test_shapenet.list')) as file:
                model_list = [line.strip().replace('/', '/') for line in file]
            part_dir = "/media/wangyida/HDD/database/shapenet/test/partial/"
            gt_dir = "/media/wangyida/HDD/database/shapenet/test/partial/"
        else:
            # with open(os.path.join('./list_pcd/valid_shapenet_eccv.list')) as file:
            # with open(os.path.join('./list_pcd/thesis_teaser_shapenet.list')) as file:
            with open(os.path.join('./list_pcd/visual_shapenet_cvpr.list')) as file:
            # with open(os.path.join('./list_pcd/valid_shapenet_lamp.list')) as file:
            # with open(os.path.join('./list_pcd/valid_shapenet.list')) as file:
                # with open(os.path.join('./list_pcd/visual_shapenet_rear.list')) as file:
                # with open(os.path.join('./list_pcd/visual_shapenet_temp.list')) as file:
                # with open(os.path.join('./list_pcd/valid_shapenet_failure.list')) as file:
                model_list = [line.strip().replace('/', '/') for line in file]
            part_dir = "/media/wangyida/HDD/database/shapenet/val/partial/"
            # part_dir = "/media/wangyida/HDD/database/shapenet/val/gt/"
            gt_dir = "/media/wangyida/HDD/database/shapenet16384/val/gt/"
            """
            part_dir = "./pcds_thesis_teaser/"
            gt_dir = "./pcds_thesis_teaser/"
            """

    with torch.no_grad():
        for i, model in enumerate(model_list):
            print(model)
            if opt.dataset == 'suncg' or opt.dataset == 'eye':
                subfold = 'all_samples'
            else:
                subfold = model[:model.rfind('/')]
            input_chosen = torch.zeros((1, int(opt.npoints[0]), 3),
                                       device='cuda')

            images_mv = []
            if opt.dataset == 'suncg' or opt.dataset == 'fusion' or opt.dataset == '3rscan' or opt.dataset == 'eye':
                if opt.dataset == '3rscan' or opt.dataset == 'eye':
                    suffix = '.ply'
                else:
                    suffix = '.pcd'
                part_tmp, part_color = read_points(
                    os.path.join(part_dir, model + suffix), opt.dataset)
                gt_tmp, gt_color = read_points(
                    os.path.join(gt_dir, model + suffix), opt.dataset)
                # part, idx_sampled = resample_pcd(part_tmp, opt.npoints * 2)
                part, idx_sampled = resample_pcd(part_tmp, int(opt.npoints[0]))
                part_seg = np.round(part_color[idx_sampled] * 11)

                gt, idx_sampled = resample_pcd(gt_tmp, int(opt.npoints[1]))
                gt_seg = np.round(gt_color[idx_sampled] * 11)
            elif opt.dataset == 'shapenet':
                start = time()
                part_tmp, part_color = read_points(
                    os.path.join(part_dir, model + '.h5'), opt.dataset)
                gt_tmp, gt_color = read_points(
                    os.path.join(gt_dir, model + '.h5'), opt.dataset)
                """
                part_tmp, part_color = read_points(
                    os.path.join(part_dir, model + '.pcd'), 'suncg')
                gt_tmp, gt_color = read_points(
                    os.path.join(gt_dir, model + '.pcd'), 'suncg')
                """
                end = time()
                print(0, end - start)
                part, idx_sampled = resample_pcd(part_tmp, int(opt.npoints[0]))
                part_seg = np.round(part_color[idx_sampled] * 11)
                gt, idx_sampled = resample_pcd(gt_tmp, int(opt.npoints[1]))
                gt_seg = np.round(gt_color[idx_sampled] * 11)
                ################

                file_views = os.listdir(
                    os.path.join("/media/wangyida/HDD/database/ShapeNet_RGBs/",
                                 '%s' % model, "rendering"))
                total_views = len(file_views)
                rendering_image_indexes = range(total_views)
                cnt_images = 0
                from other_tools.dataloader import read_images
                images_mv = read_images(model)[:, :, :, :3]
                images_mv = images_mv[:, 5: 5+128, 5: 5+128, :]
                """
                for image_idx in rendering_image_indexes:
                    if file_views[image_idx][-3:] == 'png':
                        # 16 images per object
                        num_frames = 16
                        if cnt_images == num_frames:
                            break
                        else:
                            cnt_images += 1
                        rendering_image = cv2.imread(
                            os.path.join(
                                "/media/wangyida/HDD/database/ShapeNet_RGBs/",
                                '%s' % model, "rendering",
                                file_views[image_idx]),
                            cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                        rendering_image = rendering_image[5: 5+128, 5: 5+128, :]
                        os.makedirs('./pcds/images', exist_ok=True)
                        os.makedirs('./pcds/images/' + subfold, exist_ok=True)
                        cv2.imwrite(
                            os.path.join('pcds/images', model) +
                            '-%s' % image_idx + '.png', rendering_image * 255.)
                        if len(rendering_image.shape) < 3:
                            logging.error(
                                'It seems that there is something wrong with the image file %s'
                                % (image_path))
                            sys.exit(2)
                        images_mv.append(rendering_image)
                images_mv = np.asarray(images_mv)
                """
                images_mv = torch.unsqueeze(
                    torch.tensor(images_mv[:, :, :, :3]), 0).cuda()

                ################
            part = torch.unsqueeze(part, 0).cuda()
            part_seg = torch.unsqueeze(part_seg, 0).cuda()
            gt = torch.unsqueeze(gt, 0).cuda()
            gt_seg = torch.unsqueeze(gt_seg, 0).cuda()

            # Rescale and center each point cloud
            sample_mean, sample_scale = normalize.normalize(part)
            if opt.dataset != '3rscan':
                sample_scale = torch.ones_like(sample_scale)
                sample_mean = torch.zeros_like(sample_mean)
            part = (part - sample_mean) / sample_scale
            gt = (gt - sample_mean) / sample_scale

            # output = network(part.transpose(2, 1).contiguous())
            output = network(
                part=part.transpose(2, 1), images=images_mv)

            if output['shapegf']:
                output['shapegf'] = output['shapegf'].reconstruct(part)

            if opt.dataset == '3rscan':
                part = part * sample_scale + sample_mean
                gt = gt * sample_scale + sample_mean

            if opt.dataset == 'shapenet' and complete3d_benchmark == False:
                if output['softpool']:
                    _, dist, _, _ = CD(input1=output['softpool'][0], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['msn']:
                    _, dist, _, _ = CD.forward(
                        input1=output['msn'][1], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['folding']:
                    _, dist, _, _ = CD.forward(
                        input1=output['folding'][0], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['grnet']:
                    _, dist, _, _ = CD.forward(
                        input1=output['grnet'][1], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['im_grnet']:
                    _, dist, _, _ = CD.forward(
                        input1=output['im_grnet'][1], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['shapegf']:
                    _, dist, _, _ = CD.forward(
                        input1=output['shapegf'][0], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['pcn']:
                    _, dist, _, _ = CD.forward(
                        input1=output['pcn'][1], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['disp3d']:
                    _, dist, _, _ = CD.forward(
                        input1=output['disp3d'][0], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['im_disp3d']:
                    _, dist, _, _ = CD.forward(
                        input1=output['im_disp3d'][0], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['vrcnet']:
                    _, dist, _, _ = CD.forward(
                        input1=output['vrcnet'][0], input2=gt)
                    chamfer_dist = dist.mean() * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['pointr']:
                    dist1, dist2, _, _ = CD.forward(
                        input1=output['pointr'][1], input2=gt)
                    chamfer_dist = (dist1.mean() + dist2.mean()) / 2 * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['im_pointr']:
                    dist1, dist2, _, _ = CD.forward(
                        input1=output['im_pointr'][1], input2=gt)
                    chamfer_dist = (dist1.mean() + dist2.mean()) / 2 * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['snowflake']:
                    dist1, dist2, _, _ = CD.forward(
                        input1=output['snowflake'][3], input2=gt)
                    chamfer_dist = (dist1.mean() + dist2.mean()) / 2 * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                if output['im_snowflake']:
                    dist1, dist2, _, _ = CD.forward(
                        input1=output['im_snowflake'][0][3], input2=gt)
                    chamfer_dist = (dist1.mean() + dist2.mean()) / 2 * 1e4
                    hash_tab[str(subfold)]['chamfer_dist'] += chamfer_dist
                else:
                    hash_tab[str(subfold)]['chamfer_dist'] += 0

                hash_tab[str(subfold)]['cnt'] += 1
                idx = random.randint(0, 0)
                print(
                    opt.env +
                    ' val [%d/%d]  chamfer_dist: %.2f mean chamfer_dist so far: %.2f'
                    % (i + 1, len(model_list), chamfer_dist.item(),
                       hash_tab[str(subfold)]['chamfer_dist'] /
                       hash_tab[str(subfold)]['cnt']))
            if opt.dataset == 'suncg' or opt.dataset == 'eye':
                model = 'all_samples/' + model

            # save input
            pts_coord = part[0].data.cpu()[:, 0:3]
            pts_color = colormap.colormap(
                part[0] * sample_scale.data + sample_mean.data,
                gt=gt,
                gt_seg=gt_seg,
                dataset=opt.dataset)
            """
            """
            points_save.points_save(
                points=pts_coord,
                colors=pts_color,
                root='pcds/z_input',
                child=subfold,
                pfile=model)

            # save gt
            pts_coord = gt[0].data.cpu()[:, 0:3]
            pts_color = colormap.colormap(pts_coord)
            # semantics
            pts_color = matplotlib.cm.tab10(gt_seg[0, :, 0].cpu() / 11)[:, :3]
            pts_color = colormap.colormap(
                gt[0] * sample_scale.data + sample_mean.data,
                gt=gt,
                gt_seg=gt_seg,
                dataset=opt.dataset)
            points_save.points_save(
                points=pts_coord,
                colors=pts_color,
                root='pcds/z_gt',
                child=subfold,
                pfile=model)
            
            # save a volumeitrc version 
            pcd_voxel = o3d.geometry.PointCloud()
            pcd_voxel.points = o3d.utility.Vector3dVector(pts_coord.numpy())
            pcd_voxel.colors = o3d.utility.Vector3dVector(pts_color)
            pcd_voxel = pcd_voxel.voxel_down_sample(0.1)
            points_save.points_save(
                points=np.asarray(pcd_voxel.points),
                colors=np.asarray(pcd_voxel.colors),
                root='pcds/z_gt_vox',
                child=subfold,
                pfile=model)


            if opt.dataset == 'eye':
                # save points2surf
                points2surf, points2surf_color = read_points(
                    os.path.join('./pcds_eye/points2surf/', model + '.ply'),
                    opt.dataset)
                pts_coord = points2surf[:, 0:3]
                pts_color = colormap.colormap(pts_coord)
                # semantics
                pts_color = colormap.colormap(
                    points2surf.cuda() * sample_scale.data + sample_mean.data,
                    gt=gt,
                    gt_seg=gt_seg,
                    dataset=opt.dataset,
                    with_fp=False)
                """
                """
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/points2surf',
                    child=subfold,
                    pfile=model)

                points2surf, points2surf_color = read_points(
                    os.path.join('./pcds_eye/rec_vis/', model + '.ply'),
                    opt.dataset)
                pts_coord = points2surf[:, 0:3]
                pts_color = colormap.colormap(pts_coord)
                # semantics
                pts_color = colormap.colormap(
                    points2surf.cuda() * sample_scale.data + sample_mean.data,
                    gt=gt,
                    gt_seg=gt_seg,
                    dataset=opt.dataset)
                """
                """
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/rec_vis',
                    child=subfold,
                    pfile=model)

                points2surf, points2surf_color = read_points(
                    os.path.join('./pcds_eye/eval_vis/', model + '.ply'),
                    opt.dataset)
                pts_coord = points2surf[:, 0:3]
                pts_color = colormap.colormap(pts_coord)
                # semantics
                pts_color = colormap.colormap(
                    points2surf.cuda() * sample_scale.data + sample_mean.data,
                    gt=gt,
                    gt_seg=gt_seg,
                    dataset=opt.dataset)
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/eval_vis',
                    child=subfold,
                    pfile=model)
            """
            os.makedirs('pcds/z_mesh_alpha/%s' % subfold, exist_ok=True)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.float32(pts_coord))
            pcd.colors = o3d.utility.Vector3dVector(np.float32(pts_color))
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(
                pcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, 0.037, tetra_mesh, pt_map)
            # mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
            # o3d.visualization.draw_geometries([mesh])
            o3d.io.write_triangle_mesh('pcds/z_mesh_alpha/%s.ply' % model, mesh)
            """

            # save selected points on input
            if output['softpool']:
                # save output['softpool']
                for stage in range(3):
                    pts_coord = output['softpool'][stage][0].data.cpu()[:, 0:3]
                    if stage == 2:
                        labels_for_points = label_points.label_points(
                            npoints=int(opt.npoints[1]),
                            divisions=np.max((2, opt.n_regions)))
                        maxi = labels_for_points.max()
                        pts_color = matplotlib.cm.gist_rainbow(
                            labels_for_points[0:output['softpool'][stage].
                                              size(1)] / maxi)[:, 0:3]
                        pts_color = colormap.colormap(
                            output['softpool'][stage][0] * sample_scale.data +
                            sample_mean.data,
                            gt=gt,
                            gt_seg=gt_seg,
                            dataset=opt.dataset)
                    else:
                        pts_color = colormap.colormap(
                            output['softpool'][stage][0] * sample_scale.data +
                            sample_mean.data,
                            gt=gt,
                            gt_seg=gt_seg,
                            dataset=opt.dataset)
                        """
                        """
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/softpool',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['msn']:
                # save output['msn']
                for stage in range(1, 2):
                    pts_coord = output['msn'][stage][0].data.cpu()[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu(
                        ) + sample_mean[0].data.cpu()
                    pts_coord.squeeze(0)
                    if stage == 0:
                        labels_for_points = label_points.label_points(
                            npoints=8192, divisions=16)
                        maxi = labels_for_points.max()
                        """
                        pts_color = matplotlib.cm.rainbow(
                            labels_for_points[0:output['msn'][stage].size(1)] /
                            maxi)[:, 0:3]
                        """
                        pts_color = matplotlib.cm.rainbow(
                            gt_seg[0, idx1.long(), 0].cpu() / 11)[0, :, :3]
                    else:
                        pts_color = colormap.colormap(
                            output['msn'][stage][0] * sample_scale.data +
                            sample_mean.data,
                            gt=gt,
                            gt_seg=gt_seg,
                            dataset=opt.dataset)
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/msn',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['folding']:
                # save output['folding']
                for stage in range(len(output['folding'])):
                    pts_coord = output['folding'][stage][0].data.cpu()[:, 0:3]
                    pts_color = colormap.colormap(
                        output['folding'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        dataset=opt.dataset)
                    """
                        gt=gt,
                        gt_seg=gt_seg,
                    """
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/folding',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['grnet']:
                # save output['grnet']
                for stage in range(0, 2):
                    pts_coord = output['grnet'][stage][0].data.cpu()[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu(
                        ) + sample_mean[0].data.cpu()

                    _, dist, idx1, _ = CD.forward(
                        input1=output['grnet'][stage], input2=gt)
                    pts_color = colormap.colormap(
                        output['grnet'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    """
                    """

                    chamfer_dist = dist.mean()
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/grnet',
                        child=subfold,
                        pfile=model + '-' + str(stage))
                """
                # save voxels
                os.makedirs('pcds/z_voxels/%s' % subfold, exist_ok=True)
                voxels = torch.flip(
                    output['grnet'][3][0, 0, :, :, :].transpose(2, 1), [0])
                voxels = np.array(voxels.cpu())
                import mcubes
                vertices, triangles = mcubes.marching_cubes(voxels, 0)
                pts_coord = (vertices / 64.0 - 0.5)
                pts_color = colormap.colormap(pts_coord)
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/z_voxels',
                    child=subfold,
                    pfile=model)

                # save mesh from voxels
                os.makedirs('pcds/z_mesh/%s' % subfold, exist_ok=True)
                vertices /= 64.0
                vertices -= 0.5
                # vertices[:, 2] += 0.0125
                mcubes.export_obj(vertices, triangles,
                                  'pcds/z_mesh/%s.obj' % model)
                """

            if output['im_grnet']:
                # save output['im_grnet']
                for stage in range(2):
                    pts_coord = output['im_grnet'][stage][0].data.cpu()[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu(
                        ) + sample_mean[0].data.cpu()

                    _, dist, idx1, _ = CD.forward(
                        input1=output['im_grnet'][stage], input2=gt)
                    pts_color = colormap.colormap(
                        output['im_grnet'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    """
                    """

                    chamfer_dist = dist.mean()
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/im_grnet',
                        child=subfold,
                        pfile=model + '-' + str(stage))

                # save voxels
                voxels = torch.flip(
                    output['im_grnet'][2][0, 0, :, :, :], [0])
                voxels = np.array(voxels.cpu())
                import mcubes
                vertices, triangles = mcubes.marching_cubes(voxels, 0)
                pts_coord = (vertices / 64.0 - 0.5)
                pts_color = colormap.colormap(
                    torch.unsqueeze(torch.Tensor(pts_coord).cuda(), 0),
                    gt=gt,
                    gt_seg=gt_seg,
                    dataset=opt.dataset)
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/im_grnet',
                    child=subfold,
                    pfile=model + '-' + str(2))

            if output['shapegf']:
                # save output['shapegf']
                pts_coord = output['shapegf'][0][0].data.cpu()[:, 0:3]
                pts_color = colormap.colormap(
                    output['shapegf'][0][0] * sample_scale.data +
                    sample_mean.data,
                    gt=gt,
                    gt_seg=gt_seg,
                    dataset=opt.dataset)
                """
                    gt=gt,
                    gt_seg=gt_seg,
                """
                points_save.points_save(
                    points=pts_coord,
                    colors=pts_color,
                    root='pcds/shapegf',
                    child=subfold,
                    pfile=model)
                for stage in range(0, len(output['shapegf'][1]), 1):
                    pts_coord = output['shapegf'][1][stage][0].data.cpu(
                    )[:, 0:3]
                    pts_color = matplotlib.cm.copper(
                        output['shapegf'][1][stage][0].data.cpu()[:, 1] +
                        1)[:, 0:3]
                    pts_color = colormap.colormap(
                        output['shapegf'][1][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/shapegf',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['pcn']:
                # save outpu6
                for stage in range(1, len(output['pcn'])):
                    pts_coord = output['pcn'][stage][0].data.cpu()[:, 0:3]
                    pts_color = colormap.colormap(
                        output['pcn'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/pcn',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['disp3d']:
                # save output['disp3d']
                for stage in range(len(output['disp3d'])):
                    pts_coord = output['disp3d'][stage][0].data.cpu()[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    """
                    labels_for_points = label_points.label_points(
                        npoints=int(opt.npoints[1]),
                        divisions=16)
                    maxi = labels_for_points.max()
                    pts_color = matplotlib.cm.gist_rainbow(
                        labels_for_points[0:output['disp3d'][stage].size(1)]
                        / maxi)[:, 0:3]
                    """
                    pts_color = colormap.colormap(
                        output['disp3d'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    """
                    """
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/disp3d',
                        child=subfold,
                        pfile=model + '-' + str(stage))
                    """
                    if stage == 0:
                        for part in range(4096 // 64):
                            points_save.points_save(
                                points=pts_coord[64*part:64*part+64],
                                colors=pts_color[64*part:64*part+64],
                                root='pcds/disp3d',
                                child=subfold,
                                pfile=model + '-' + str(stage) + '-' + str(part))
                    """

            if output['im_disp3d']:
                # save output['disp3d']
                for stage in range(len(output['im_disp3d'])):
                    pts_coord = output['im_disp3d'][stage][0].data.cpu(
                    )[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    """
                    labels_for_points = label_points.label_points(
                        npoints=int(opt.npoints[1]),
                        divisions=16)
                    maxi = labels_for_points.max()
                    pts_color = matplotlib.cm.gist_rainbow(
                        labels_for_points[0:output['disp3d'][stage].size(1)]
                        / maxi)[:, 0:3]
                    """
                    pts_color = colormap.colormap(
                        output['im_disp3d'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    """
                    """
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/im_disp3d',
                        child=subfold,
                        pfile=model + '-' + str(stage))
                    """
                    if stage == 0:
                        for part in range(4096 // 64):
                            points_save.points_save(
                                points=pts_coord[64*part:64*part+64],
                                colors=pts_color[64*part:64*part+64],
                                root='pcds/disp3d',
                                child=subfold,
                                pfile=model + '-' + str(stage) + '-' + str(part))
                    """

            if output['vrcnet']:
                # for stage in range(len(output['vrcnet'])):
                for stage in range(2):
                    pts_coord = output['vrcnet'][stage][0].data.cpu()[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    pts_color = colormap.colormap(
                        output['vrcnet'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        dataset=opt.dataset)
                    """
                        gt=gt,
                        gt_seg=gt_seg,
                    """
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/vrcnet',
                        child=subfold,
                        pfile=model + '-' + str(stage))

                # Submission
                if opt.dataset == 'shapenet' and complete3d_benchmark == True:
                    os.makedirs('benchmark', exist_ok=True)
                    os.makedirs('benchmark/' + subfold, exist_ok=True)
                    with h5py.File('benchmark/' + model + '.h5', "w") as f:
                        f.create_dataset("data", data=np.float32(pts_coord))

            if output['pointr']:
                for stage in range(len(output['pointr'])):
                    # for stage in range(2):
                    pts_coord = output['pointr'][stage][0].data.cpu()[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    pts_color = colormap.colormap(
                        output['pointr'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    if stage == 0:
                        """
                        npoint = output['pointr'][stage].size(1)
                        labels_for_points = label_points.label_points(
                            npoints=npoint,
                            divisions=npoint)
                        maxi = labels_for_points.max()
                        pts_color = matplotlib.cm.cool(
                            labels_for_points[0:npoint]
                            / maxi)[:, 0:3]
                        """
                        pts_coord = pts_coord[:224, :]
                        pts_color = pts_color[:224, :]
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/pointr',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['im_pointr']:
                # for stage in range(3, len(output['im_pointr'])):
                for stage in range(len(output['im_pointr'])):
                    pts_coord = output['im_pointr'][stage][0].data.cpu(
                    )[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    pts_color = colormap.colormap(
                        output['im_pointr'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/im_pointr',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            if output['snowflake']:
                for stage in range(3, len(output['snowflake'])):
                    # for stage in range(2):
                    pts_coord = output['snowflake'][stage][0].data.cpu(
                    )[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    pts_color = colormap.colormap(
                        output['snowflake'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    """
                    """
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/snowflake',
                        child=subfold,
                        pfile=model + '-' + str(stage))

            """
            if output['im_snowflake']:
                for stage in range(len(output['im_snowflake'])):
                    # for stage in range(2):
                    pts_coord = output['im_snowflake'][stage][0].data.cpu(
                    )[:, 0:3]
                    if opt.dataset == '3rscan':
                        pts_coord *= sample_scale[0].data.cpu()
                        pts_coord += sample_mean[0].data.cpu()
                    pts_color = colormap.colormap(
                        output['im_snowflake'][stage][0] * sample_scale.data +
                        sample_mean.data,
                        gt=gt,
                        gt_seg=gt_seg,
                        dataset=opt.dataset)
                    points_save.points_save(
                        points=pts_coord,
                        colors=pts_color,
                        root='pcds/im_snowflake',
                        child=subfold,
                        pfile=model + '-' + str(stage))
            """

            if output['im_snowflake']:
                for source in range(len(output['im_snowflake'])):
                    for stage in range(len(output['im_snowflake'][source])):
                        pts_coord = output['im_snowflake'][source][stage][0].data.cpu(
                        )[:, 0:3]
                        if opt.dataset == '3rscan':
                            pts_coord *= sample_scale[0].data.cpu()
                            pts_coord += sample_mean[0].data.cpu()
                        pts_color = colormap.colormap(
                            output['im_snowflake'][source][stage][0] * sample_scale.data +
                            sample_mean.data,
                            gt=gt,
                            gt_seg=gt_seg,
                            dataset=opt.dataset)
                        points_save.points_save(
                            points=pts_coord,
                            colors=pts_color,
                            root='pcds/im_snowflake',
                            child=subfold,
                            pfile=model + '-' + str(source) + '-' + str(stage))
        if opt.dataset == 'shapenet' and complete3d_benchmark == False:
            names_categories = [
                '04530566', '02933112', '04379243', '02691156', '02958343',
                '03001627', '04256520', '03636649'
            ]
            min_samples = 1
            for i in names_categories:
                hash_tab
                if hash_tab[i]['cnt'] > 0:
                    print('%s chamfer_dist: %.2f' %
                          (hash_tab[i]['name'],
                           hash_tab[i]['chamfer_dist'] / hash_tab[i]['cnt']))


if __name__ == "__main__":
    main()
