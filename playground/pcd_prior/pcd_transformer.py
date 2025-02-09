import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from playground.pcd_prior.utils import is_initial_image, quaternion_to_rotation_matrix


class PointCloudTransformer:
    blender_transforms_path: str
    colmap_reconstruction_path: str

    def __init__(self, blender_transforms_path: str, colmap_reconstruction_path: str):
        self.blender_transforms_path = blender_transforms_path
        self.colmap_reconstruction_path = colmap_reconstruction_path

        self.blender_camera_centers = self._get_blender_camera_centers()
        self.colmap_camera_centers = self._get_colmap_camera_centers()

        self.test_rigid_transform()

    def _get_blender_camera_centers(self) -> dict:
        with open(self.blender_transforms_path, 'r') as file:
            transforms = json.load(file)

            camera_centers = dict()
            for frame in transforms['frames']:
                if not is_initial_image(frame['file_path']):
                    continue
                transform_matrix = np.array(frame['transform_matrix'])
                translation_vector = transform_matrix[0:3, 3]  # we don't need the rotation matrix
                file_name = frame['file_path'].split("/")[-1]
                camera_centers[file_name] = translation_vector

            return camera_centers
    def _get_colmap_camera_centers(self) -> dict:
        with open(self.colmap_reconstruction_path, 'r') as file:
            lines = file.readlines()
        
        camera_centers = dict()
        for i in range(0, len(lines), 2):
            line = lines[i].strip().split()
            if line[0] == '#':  # skip comments
                continue

            # We don't actually need the rotation matrix, this is just for style points
            qw, qx, qy, qz = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)

            tx, ty, tz = float(line[5]), float(line[6]), float(line[7])
            translation_vector = np.array([tx, ty, tz])
            file_name = line[9]
            camera_centers[file_name] = np.dot(-rotation_matrix.T, translation_vector)

        return camera_centers
    
    @staticmethod
    def apply_transformation(A, R, t):
        return (R @ A.T).T + t
    
    def test_rigid_transform(self):
        """
        Unit test for compute_rigid_transform and apply_transformation.
        """
        # Define two simple corresponding point clouds
        A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Original points
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation about Z-axis
        t_true = np.array([1, 2, 3])  # Translation vector

        # Apply the known transformation to create B
        B = (R_true @ A.T).T + t_true

        # Compute the transformation from A to B
        R_computed, t_computed = self.compute_rigid_transform(A, B)

        # Apply computed transformation to A
        A_transformed = self.apply_transformation(A, R_computed, t_computed)

        # Check if transformed A matches B
        assert np.allclose(A_transformed, B, atol=1e-6), "Test failed: Transformed points do not match target!"

        print("Test passed: Transformed points align with target!")
    
    def transform_blender_to_colmap(self):
        # Filter centers by intersecting dicts
        common_keys = set(self.blender_camera_centers.keys()).intersection(self.colmap_camera_centers.keys())

        # Sort common keys to maintain consistent order
        sorted_keys = sorted(common_keys)

        # Extract and sort values based on the sorted common keys
        blender_points = np.array([self.blender_camera_centers[k] for k in sorted_keys])
        colmap_points = np.array([self.colmap_camera_centers[k] for k in sorted_keys])

        R, t = self.compute_rigid_transform(blender_points, colmap_points)
        self.R = R
        self.t = t

        return R, t
    
    def transform_colmap_to_blender(self):
        R, t = self.transform_blender_to_colmap()
    
        # Compute the inverse transformation
        R_inv = R.T  # Since R is a rotation matrix, its inverse is its transpose
        t_inv = -R_inv @ t  # Inverted translation

        self.R = R_inv
        self.t = t_inv

        return R_inv, t_inv
    
    def visualize_transformation(self):
        common_keys = set(self.blender_camera_centers.keys()).intersection(self.colmap_camera_centers.keys())
        sorted_keys = sorted(common_keys)

        blender_points = np.array([self.blender_camera_centers[k] for k in sorted_keys])
        colmap_points = np.array([self.colmap_camera_centers[k] for k in sorted_keys])

        colmap_transformed = self.apply_transformation(colmap_points, self.R, self.t)

        mse = np.mean(np.abs(blender_points - colmap_transformed), axis=0)
        print(f"MSE: {mse}")
        
        self.visualize_point_clouds([blender_points, colmap_points, colmap_transformed], ['Blender', 'Colmap', 'Colmap Transformed'])
    
    @staticmethod
    def compute_rigid_transform(A, B):
        """
        Compute the optimal rigid transformation (R, t) that aligns point cloud A to B.
        :param A: (N, 3) array of 3D points in the source point cloud.
        :param B: (N, 3) array of corresponding 3D points in the target point cloud.
        :return: Rotation matrix R (3x3), translation vector t (3x1)
        """
        assert A.shape == B.shape, "Point clouds must have the same shape"

        # Compute centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # Center the points (subtract centroids)
        A_centered = A - centroid_A
        B_centered = B - centroid_B

        # Compute covariance matrix
        H = A_centered.T @ B_centered

        # Compute SVD
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure a proper rotation matrix (det(R) should be 1, not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation vector
        t = centroid_B - R @ centroid_A

        return R, t
    
    @staticmethod
    def visualize_point_clouds(point_clouds: iter, names: iter) -> None:
        """
        Visualizes multiple 3D point clouds with different colors.

        Args:
            point_clouds (iterable of np.ndarray): Iterable containing point clouds of shape (n, 3).
            names (iterable of str): Iterable containing names corresponding to the point clouds.
        """
        point_clouds = list(point_clouds)
        names = list(names)
        
        assert len(point_clouds) == len(names), "Point clouds and names must have the same length."
        
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # Extendable color list
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i, (pc, name) in enumerate(zip(point_clouds, names)):
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors[i % len(colors)], label=name)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()