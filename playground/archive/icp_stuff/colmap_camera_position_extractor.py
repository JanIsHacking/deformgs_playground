import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

class ColmapCameraPositionExtractor:
    """
    A class to extract camera positions from a COLMAP reconstruction's `images.txt` file,
    convert them into 3D vectors, and visualize the camera positions in 3D space.

    Attributes:
        file_path (str): The path to the `images.txt` file containing the camera data.
    """
    _camera_centers: np.ndarray

    def __init__(self, file_path: str):
        """
        Initializes the ColmapCameraPositionExtractor with the path to the `images.txt` file.

        Args:
            file_path (str): The path to the `images.txt` file.
        """
        self.file_path = file_path

    def quaternion_to_rotation_matrix(self, qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """
        Converts a quaternion (qw, qx, qy, qz) into a 3x3 rotation matrix.

        Args:
            qw (float): The scalar part of the quaternion.
            qx (float): The x component of the quaternion.
            qy (float): The y component of the quaternion.
            qz (float): The z component of the quaternion.

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        # Manually compute the rotation matrix from the quaternion components
        norm = qw*qw + qx*qx + qy*qy + qz*qz
        s = 2.0 / norm  # Scaling factor to ensure proper rotation matrix
        
        # Compute the rotation matrix elements
        xx = qx * qx * s
        yy = qy * qy * s
        zz = qz * qz * s
        xy = qx * qy * s
        xz = qx * qz * s
        yz = qy * qz * s
        wx = qw * qx * s
        wy = qw * qy * s
        wz = qw * qz * s
        
        # Rotation matrix
        rotation_matrix = np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)]
        ])
        
        return rotation_matrix

    def read_camera_data(self) -> np.ndarray:
        """
        Reads the `images.txt` file and extracts the camera positions and rotations.

        Each line in the `images.txt` file contains camera rotation (as a quaternion) and
        translation data (camera position). This method computes the camera center in
        3D space based on the rotation and translation data.

        Returns:
            np.ndarray: A numpy array of shape (n, 3) representing camera centers in 3D space.
        """
        camera_centers = []
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 2):
            line = lines[i].strip().split()  # Start from line 2 and skip every 2 lines (image data)
            if line[0] == '#':
                continue
            # Parse the quaternion and translation values
            image_id = int(line[0])
            qw, qx, qy, qz = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            tx, ty, tz = float(line[5]), float(line[6]), float(line[7])
            file_name = line[9]

            # Convert quaternion to rotation matrix
            rotation_matrix = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)
            
            # Calculate the camera center (translation vector applied to the origin)
            camera_center = np.dot(-rotation_matrix.T, np.array([tx, ty, tz]))  # Apply the inverse rotation
            camera_centers.append(camera_center)

        return np.array(camera_centers)

    def extract(self) -> np.ndarray:
        """
        Extracts camera positions from the `images.txt` file and visualizes them in 3D.

        Returns:
            np.ndarray: A numpy array containing all the camera centers in 3D space.
        """
        self._camera_centers = self.read_camera_data()
        return self._camera_centers
