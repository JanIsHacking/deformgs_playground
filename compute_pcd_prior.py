import os
import numpy as np
import matplotlib.pyplot as plt
from playground.pcd_prior.pcd_transformer import PointCloudTransformer
from playground.pcd_prior.utils import load_colmap_point_cloud, remove_outliers
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement


def visualize_point_cloud(point_cloud: BasicPointCloud):
    # Sample 1/4 of the points randomly
    num_points = len(point_cloud.points)
    sample_size = num_points // 10
    indices = np.random.choice(num_points, sample_size, replace=False)
    
    points = point_cloud.points[indices]
    colors = point_cloud.colors[indices]
    normals = point_cloud.normals[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points with colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=5)

    # Plot normals as arrows
    for i in range(len(points)):
        ax.quiver(points[i, 0], points[i, 1], points[i, 2],
                  normals[i, 0], normals[i, 1], normals[i, 2], length=0.1, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def storePly(path, xyz, rgb, normals):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    rgb = (rgb * 255).astype(np.uint8)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def main():
    pcd_transformer = PointCloudTransformer(
        blender_transforms_path='data/synthetic/scene_1/transforms_train.json',
        colmap_reconstruction_path='data/synthetic/scene_1/colmap/images.txt'
    )
    R, t = pcd_transformer.transform_blender_to_colmap()

    # pcd_transformer.visualize_transformation()

    # Use the transformation to transform the colmap point cloud of the scene into the blender world frame
    colmap_point_cloud = load_colmap_point_cloud('data/synthetic/scene_1/colmap/points3D.txt')
    colmap_point_cloud = remove_outliers(colmap_point_cloud, radius=0.05)

    colmap_points_transformed = PointCloudTransformer.apply_transformation(colmap_point_cloud.points, R, t)
    colmap_point_cloud = BasicPointCloud(colmap_points_transformed, colmap_point_cloud.colors, colmap_point_cloud.normals)

    visualize_point_cloud(colmap_point_cloud)

    # storePly('data/synthetic/scene_1/initial_pcd.ply', colmap_point_cloud.points, colmap_point_cloud.colors, colmap_point_cloud.normals)


if __name__ == '__main__':
    main()
