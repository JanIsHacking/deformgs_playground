import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scene.gaussian_model import BasicPointCloud

def compute_normals(point_cloud, k_neighbors=100):
    """
    Computes normals for each point in the point cloud using PCA.
    
    Parameters:
        point_cloud (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud.
        k_neighbors (int): The number of neighbors to consider for each point to compute the normal.
    
    Returns:
        normals (numpy.ndarray): A numpy array of shape (N, 3) containing the normal vectors for each point.
    """
    # Initialize NearestNeighbors model to find the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(point_cloud)
    
    # Find the indices of the nearest neighbors
    _, indices = nbrs.kneighbors(point_cloud)
    
    # Array to store the normals
    normals = np.zeros_like(point_cloud)
    
    # Loop over each point in the point cloud
    for i, idx in enumerate(indices):
        # Get the neighbors' coordinates
        neighbors = point_cloud[idx]
        
        # Perform PCA to find the principal components
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        
        # The normal corresponds to the smallest eigenvalue (last component)
        normal = pca.components_[-1]
        
        # Store the normal for the current point
        normals[i] = normal
    
    return normals

def remove_outliers(point_cloud: BasicPointCloud, radius=0.1):
    points = point_cloud.points
    colors = point_cloud.colors
    normals = point_cloud.normals
    
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, r=radius)
    
    # Find indices of points that have no other neighbors (outliers)
    keep_indices = np.array([len(n) > 1 for n in neighbors])
    
    # Filter points, colors, and normals using the keep_indices
    filtered_points = points[keep_indices]
    filtered_colors = colors[keep_indices]
    filtered_normals = normals[keep_indices]
    
    # Return the new BasicPointCloud with filtered data
    return BasicPointCloud(filtered_points, filtered_colors, filtered_normals)

def load_colmap_point_cloud(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        points = []
        colors = []
        normals = []
        for line in lines:
            if line.startswith('#'):
                continue
            points.append(line.split()[1:4])
            colors.append(line.split()[4:7])
        normals = compute_normals(np.array(points))
        return BasicPointCloud(np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32) / 255.0, normals)


def is_initial_image(image_path: str):
    return image_path[-6:] == '_0.png'

def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
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
