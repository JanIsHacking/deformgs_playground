import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List


class TransformsCameraPositionExtractor:
    """
    A class to extract camera positions from a JSON file containing transformation matrices,
    compute camera centers, and visualize them as a 3D point cloud.

    Attributes:
        file_path (str): The path to the JSON file containing transformation matrices for the camera frames.
    """
    _camera_centers: np.ndarray

    def __init__(self, file_path: str):
        """
        Initializes the TransformsCameraPositionExtractor with the path to the JSON file.

        Args:
            file_path (str): The path to the JSON file containing camera transformation data.
        """
        self.file_path = file_path

    def _load_json(self) -> Dict[str, Any]:
        """
        Loads the JSON data from the provided file path.

        Returns:
            Dict[str, Any]: The loaded JSON data.
        """
        with open(self.file_path, 'r') as file:
            return json.load(file)

    def _extract_camera_center(self, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Extracts the camera center from the transformation matrix by calculating the inverse
        of the rotation matrix and applying it to the translation vector.

        Args:
            transform_matrix (np.ndarray): A 4x4 transformation matrix for the camera frame.

        Returns:
            np.ndarray: The 3D camera center position.
        """
        # Extract the rotation matrix (top-left 3x3) and the translation vector (last column)
        rotation_matrix = transform_matrix[0:3, 0:3]
        translation_vector = transform_matrix[0:3, 3]
        camera_center = np.dot(rotation_matrix, np.zeros(3)) + translation_vector
        return camera_center

    def _create_point_cloud(self, json_data: Dict[str, Any]) -> np.ndarray:
        """
        Creates a point cloud of camera centers by extracting them from each frame's transformation matrix.

        Args:
            json_data (Dict[str, Any]): The loaded JSON data containing camera frames and their transformations.

        Returns:
            np.ndarray: A numpy array containing all the camera centers in 3D space.
        """
        camera_centers = []

        for frame in json_data['frames']:
            if frame['file_path'][-6:] == '_0.png':  # Ensure to only use frames that match the condition
                transform_matrix = np.array(frame['transform_matrix'])
                camera_center = self._extract_camera_center(transform_matrix)
                camera_centers.append(camera_center)
        
        return np.array(camera_centers)

    def visualize(self) -> None:
        """
        Visualizes the camera centers in a 3D scatter plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self._camera_centers[:, 0], self._camera_centers[:, 1], self._camera_centers[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()

    def extract(self) -> np.ndarray:
        """
        Extracts camera positions from the JSON file.

        Returns:
            np.ndarray: A numpy array containing all the camera centers in 3D space.
        """
        json_data = self._load_json()
        self._camera_centers = self._create_point_cloud(json_data)
        return self._camera_centers
