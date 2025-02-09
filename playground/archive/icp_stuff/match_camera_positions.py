from matplotlib import pyplot as plt
import numpy as np
from colmap_camera_position_extractor import ColmapCameraPositionExtractor
from point_cloud_transformer import PointCloudTransformer
from transforms_camera_position_extractor import TransformsCameraPositionExtractor

def visualize_pcd(pcd: np.ndarray):
    """
    Visualizes the point cloud in a 3D scatter plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


if __name__ == '__main__':
    file_path = 'data/synthetic/scene_1/transforms_train.json'  # Change this to the path of your transforms.json file
    transforms_camera_extractor = TransformsCameraPositionExtractor(file_path)
    transforms_camera_centers = transforms_camera_extractor.extract()
    transforms_camera_centers = transforms_camera_centers @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) + np.array([0, 0, 2.5])

    file_path = 'data/synthetic/scene_1/pcd-init/colmap/images.txt'  # Change this to the path of your images.txt file
    colmap_camera_extractor = ColmapCameraPositionExtractor(file_path)
    colmap_camera_centers = colmap_camera_extractor.extract()

    visualize_pcd(np.vstack([transforms_camera_centers, colmap_camera_centers]))

    transformer = PointCloudTransformer()

    # Compute the transformation
    transformation_matrix, fitness_score = transformer.compute_transformation(transforms_camera_centers, colmap_camera_centers)
