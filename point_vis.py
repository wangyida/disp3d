# examples/Python/Basic/pointcloud.py

import os
import argparse
import csv
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def custom_draw_geometry_with_rotation(pcd):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        # param = o3d.io.read_pinhole_camera_trajectory("/Users/yidawang/Documents/gitfarm/sem-pts-com/camera_trajectory.json").parameters[0]
        # ctr.convert_from_pinhole_camera_parameters(param)
        # ctr.translate(0.5,0.5)
        # ctr.scale(2)
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)


def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "/Users/yidawang/Documents/gitfarm/sem-pts-com/camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    if not os.path.exists("../../TestData/image/"):
        os.makedirs("../../TestData/image/")
    if not os.path.exists("../../TestData/depth/"):
        os.makedirs("../../TestData/depth/")

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            """
            plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
                    np.asarray(depth), dpi = 1)
            plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
                    np.asarray(image), dpi = 1)
            """
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(
        "/Users/yidawang/Documents/gitfarm/sem-pts-com/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-n',
        action="store",
        type=int,
        dest="numbers",
        default=5,
        help='Number of samples')
    parser.add_argument(
        '-f',
        action="store",
        dest="file_path",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l',
        action="store",
        dest="files_list",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l2',
        action="store",
        dest="files_list2",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l3',
        action="store",
        dest="files_list3",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l4',
        action="store",
        dest="files_list4",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l5',
        action="store",
        dest="files_list5",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l6',
        action="store",
        dest="files_list6",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l7',
        action="store",
        dest="files_list7",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l8',
        action="store",
        dest="files_list8",
        default="",
        help='Destination for storing results')
    parser.add_argument(
        '-l9',
        action="store",
        dest="files_list9",
        default="",
        help='Destination for storing results')
    parser.print_help()
    results = parser.parse_args()
    if results.file_path != '':
        pcd = o3d.io.read_point_cloud(results.file_path)
        voxels = pcd.voxel_down_sample(voxel_size=0.02)
        activ = ((np.asarray(voxels.points) + 0.5) * 50).astype(int)
        converted_vox = np.zeros(shape=(64, 64, 64))
        # converted_vox[activ[:,0], activ[:,1], activ[:,2]] = 1
        # np.save('voxels/1.npy', converted_vox)

        # pcd = o3d.io.read_triangle_mesh(results.file_path)
        # pcd.compute_vertex_normals()
        print("Load a ply point cloud, print it, and render it")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05, max_nn=64))
        cam_view = "/Users/yidawang/Documents/gitfarm/sem-pts-com/qualitative_lists/render_plane.json"
        print("Recompute the normal of the downsampled point cloud")
        # save_view_point(pcd, cam_view)
        load_view_point(pcd, cam_view)

        radii = [0.005, 0.01, 0.02, 0.04]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        o3d.visualization.draw_geometries([pcd, rec_mesh])

        print("Statistical oulier removal")
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=4.0)
        # load_view_point(cl, cam_view)
        # display_inlier_outlier(pcd, ind)

        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        # o3d.visualization.draw_geometries([mesh])
        custom_draw_geometry_with_rotation(pcd)

        # custom_draw_geometry_with_rotation(pcd)
        print("Downsample the point cloud with a voxel of 0.02")
        # pcd = pcd.voxel_down_sample(voxel_size=0.02)
        # suncg
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.025)
        # shapenet
        colors = (np.asarray(pcd.points) + 0.1) * 1.5
        colors[:, 0] = colors[:, 1] - np.min(colors[:, 1])
        colors[:, 1] = colors[:, 1] - np.max(colors[:, 1])
        colors[:, 2] = 0.5 - colors[:, 1]
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=0.015625)
        o3d.visualization.draw_geometries([voxel_grid],
                                          mesh_show_wireframe=True)
        """
        print("Radius oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=0.5)
        o3d.visualization.draw_geometries([cl])
        display_inlier_outlier(pcd, ind)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=64))
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        o3d.visualization.draw_geometries([mesh])
        """
    elif results.files_list != '':
        # Get the number of validation samples
        """
        with open(results.files_list, "r") as f:
            points_all = np.zeros((1, 3))
            colors_all = np.zeros((1, 3))
            reader = csv.reader(f)
            data = list(reader)
            for i in range(results.numbers):
                pcd = o3d.io.read_point_cloud(data[i][0])
                if i == 1:
                    pcd, ind = pcd.remove_statistical_outlier(
                        nb_neighbors=8, std_ratio=2.0)
                npy_points = np.asarray(pcd.points)
                npy_points[:, 0] += (i * 1.002)
                npy_colors = np.asarray(pcd.colors)
                points_all = np.concatenate((points_all, npy_points), 0)
                colors_all = np.concatenate((colors_all, npy_colors), 0)
        """
        with open(results.files_list, "r") as f:
            points_all = np.zeros((1, 3))
            colors_all = np.zeros((1, 3))
            reader = csv.reader(f)
            data = list(reader)
            translation = list(
                csv.reader(
                    open('/Users/yidawang/Downloads/scene0050_02_location.txt',
                         'r'),
                    delimiter=" "))
            for i in range(results.numbers):
                pcd = o3d.io.read_point_cloud(data[i][0])
                # pcd = pcd.voxel_down_sample(voxel_size=1)
                npy_points = np.asarray(pcd.points)
                x_trans = float(translation[i][1])
                y_trans = float(translation[i][3])
                """
                # For scene completion with given translations
                npy_points[:, 0] = npy_points[:, 0]*0.05+y_trans
                npy_points[:, 1] = npy_points[:, 1]*0.05
                npy_points[:, 2] = npy_points[:, 2]*0.05+x_trans
                """
                npy_points[:, 1] += (i * .9002)
                npy_colors = np.asarray(pcd.colors)
                points_all = np.concatenate((points_all, npy_points), 0)
                colors_all = np.concatenate((colors_all, npy_colors), 0)

                pcd.points = o3d.utility.Vector3dVector(points_all[:, 0:3])
                pcd.colors = o3d.utility.Vector3dVector(colors_all[:, 0:3])
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=32))

            o3d.visualization.draw_geometries([pcd])
        if results.files_list2 != '':
            with open(results.files_list2, "r") as f:
                points_all2 = np.zeros((1, 3))
                colors_all2 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += .7002
                    # npy_points[:, 1] += .4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all2 = np.concatenate((points_all2, npy_points), 0)
                    colors_all2 = np.concatenate((colors_all2, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all2), 0)
                    colors_all = np.concatenate((colors_all, colors_all2), 0)
        if results.files_list3 != '':
            with open(results.files_list3, "r") as f:
                points_all3 = np.zeros((1, 3))
                colors_all3 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 2 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all3 = np.concatenate((points_all3, npy_points), 0)
                    colors_all3 = np.concatenate((colors_all3, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all3), 0)
                    colors_all = np.concatenate((colors_all, colors_all3), 0)
        if results.files_list4 != '':
            with open(results.files_list4, "r") as f:
                points_all4 = np.zeros((1, 3))
                colors_all4 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 3 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all4 = np.concatenate((points_all4, npy_points), 0)
                    colors_all4 = np.concatenate((colors_all4, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all4), 0)
                    colors_all = np.concatenate((colors_all, colors_all4), 0)
        if results.files_list5 != '':
            with open(results.files_list5, "r") as f:
                points_all5 = np.zeros((1, 3))
                colors_all5 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 4 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all5 = np.concatenate((points_all5, npy_points), 0)
                    colors_all5 = np.concatenate((colors_all5, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all5), 0)
                    colors_all = np.concatenate((colors_all, colors_all5), 0)
        if results.files_list6 != '':
            with open(results.files_list6, "r") as f:
                points_all6 = np.zeros((1, 3))
                colors_all6 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 5 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all6 = np.concatenate((points_all6, npy_points), 0)
                    colors_all6 = np.concatenate((colors_all6, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all6), 0)
                    colors_all = np.concatenate((colors_all, colors_all6), 0)
        if results.files_list7 != '':
            with open(results.files_list7, "r") as f:
                points_all7 = np.zeros((1, 3))
                colors_all7 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 6 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all7 = np.concatenate((points_all7, npy_points), 0)
                    colors_all7 = np.concatenate((colors_all7, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all7), 0)
                    colors_all = np.concatenate((colors_all, colors_all7), 0)
        if results.files_list8 != '':
            with open(results.files_list8, "r") as f:
                points_all8 = np.zeros((1, 3))
                colors_all8 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 7 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all8 = np.concatenate((points_all8, npy_points), 0)
                    colors_all8 = np.concatenate((colors_all8, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all8), 0)
                    colors_all = np.concatenate((colors_all, colors_all8), 0)
        if results.files_list9 != '':
            with open(results.files_list9, "r") as f:
                points_all9 = np.zeros((1, 3))
                colors_all9 = np.zeros((1, 3))
                reader = csv.reader(f)
                data = list(reader)
                for i in range(results.numbers):
                    pcd = o3d.io.read_point_cloud(data[i][0])
                    npy_points = np.asarray(pcd.points)
                    npy_points[:, 0] += (i * .9002)
                    npy_points[:, 2] += 8 * .7002
                    # npy_points[:, 1] += 2*.4002
                    npy_colors = np.asarray(pcd.colors)
                    points_all9 = np.concatenate((points_all9, npy_points), 0)
                    colors_all9 = np.concatenate((colors_all9, npy_colors), 0)
                    print(pcd)
                    points_all = np.concatenate((points_all, points_all9), 0)
                    colors_all = np.concatenate((colors_all, colors_all9), 0)
        pcd.points = o3d.utility.Vector3dVector(points_all[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(colors_all[:, 0:3])
        # o3d.visualization.draw_geometries([pcd])

        print("Recompute the normal of the downsampled point cloud")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=32))

        o3d.visualization.draw_geometries([pcd])
        custom_draw_geometry_with_rotation(pcd)

        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.0)
        o3d.visualization.draw_geometries([cl])
        display_inlier_outlier(pcd, ind)
