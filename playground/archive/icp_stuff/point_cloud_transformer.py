import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class PointCloudTransformer:
    def __init__(self):
        pass

    def compute_transformation(self, source, target, max_iterations=2000, threshold=3):
        """
        Compute the transformation matrix to align cloud_in with cloud_tr using ICP.

        Args:
            source (np.ndarray): Input point cloud (source).
            target (np.ndarray): Target point cloud (aligned).
            max_iterations (int): Maximum number of ICP iterations.
            threshold (float): Distance threshold for ICP convergence.
        
        Returns:
            transformation_matrix (np.ndarray): The 4x4 transformation matrix.
            fitness_score (float): The fitness score of the alignment.
        """
        # Ensure the point clouds are properly loaded as open3d.geometry.PointCloud
        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()
        
        source_pcd.points = o3d.utility.Vector3dVector(source)
        target_pcd.points = o3d.utility.Vector3dVector(target)
        
        # ICP registration
        threshold = 0.02  # Adjust the threshold based on the noise and point density
        trans_init = np.eye(4)  # Initial transformation matrix

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )

        print("Transformation is:")
        print(reg_p2p.transformation)

        print("Fitness is:")
        print(reg_p2p.fitness)
        
        return reg_p2p.transformation, reg_p2p.fitness

    def visualize_registration(self, cloud_in, cloud_tr, transformation_matrix):
        """
        Visualizes the point clouds before and after registration.

        Args:
            cloud_in (np.ndarray): Input point cloud (source).
            cloud_tr (np.ndarray): Target point cloud (aligned).
            transformation_matrix (np.ndarray): The transformation matrix.
        """
        # Convert numpy arrays to Open3D point clouds
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()

        source.points = o3d.utility.Vector3dVector(cloud_in)
        target.points = o3d.utility.Vector3dVector(cloud_tr)

        # Transform the input cloud using the computed transformation matrix
        source.transform(transformation_matrix)

        # Visualize the point clouds
        o3d.visualization.draw_geometries([source, target],
                                          window_name="ICP Point Cloud Registration",
                                          width=800, height=600)

