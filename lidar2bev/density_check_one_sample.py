import numpy as np
import glob
from tqdm import tqdm

def load_kitti_bin(bin_path: str) -> np.ndarray:
    """
    Loads a KITTI .bin Point Cloud and returns only x, y, z coordinates.
    """
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # [x, y, z]

def compute_density(point_cloud: np.ndarray, cell_size: float = 0.1) -> np.ndarray:
    """
    Divides a point cloud into a regular grid and computes point density.

    Parameters:
    - point_cloud: A numpy array of shape [n, 3] containing x, y, z coordinates.
    - cell_size: The size of each grid cell in meters (default is 0.1m).

    Returns:
    - density_map: A 2D numpy array where each cell contains the count of points.
    - total_cells: The total number of grid cells.
    """
    x, y = point_cloud[:, 0], point_cloud[:, 1]

    # Grid bounds from cloud
    x_min, y_min = np.min(x), np.min(y)
    x_max, y_max = np.max(x), np.max(y)

    # Calculate cell indices in x and y directions
    x_indices = ((x - x_min) / cell_size).astype(int)
    y_indices = ((y - y_min) / cell_size).astype(int)

    # Create a grid to count points in each cell
    grid_x = int((x_max - x_min) / cell_size) + 1 # to ensure that the maximum value is also covered
    grid_y = int((y_max - y_min) / cell_size) + 1

    # Aggregate points into grid cells
    density_map = np.zeros((grid_y, grid_x), dtype=np.int32)
    np.add.at(density_map, (y_indices, x_indices), 1)

    # Calculate total 0.1x0.1m cells in scene
    total_cells = grid_x * grid_y
    
    return density_map, total_cells

def process_bin_files(file_path: str) -> tuple:
    """
    Processes a .bin file and computes density metrics.

    Parameters:
    - file_path: Path to the .bin file.

    Returns:
    - density_map: 2D array of density values (points per grid cell).
    - max_density: Maximum density value among all grid cells.
    - avg_density: Average density value among all grid cells.
    - total_cells: Total number of grid cells.
    """
        
    print("""
    This program analyzes LiDAR point clouds from KITTI data and computes point density per grid cell.
    Steps performed by the script:
    1. Load a LiDAR point cloud from a `.bin` file.
    2. Divide the point cloud into a regular grid (default cell size: 0.1m x 0.1m).
    3. Count the number of points in each grid cell to compute the density.
    4. Output the total number of grid cells, and the maximum and average point density per cell.
    """)

    # Load point cloud from .bin file
    print(f"Processing file: {file_path}")
    point_cloud = load_kitti_bin(file_path)

    # Compute the density map for the current file
    density_map, t_cells = compute_density(point_cloud)
    density_flat = density_map.flatten()

    # Calculate the max and avg densities
    max_density = np.max(density_flat)
    avg_density = np.mean(density_flat)

    return density_map, max_density, avg_density, t_cells


if __name__ == "__main__":
    bin_file = "/home/rlab10/Downloads/training/velodyne/000000.bin"
    
    densities, max_density, avg_density, t_cells = process_bin_files(bin_file)

    print(f'Total grid cells: {t_cells}')
    print(f"Maximum Density: {max_density:.2f} Points/0.1m² cell")
    print(f"Average Density: {avg_density:.2f} Points/0.1m² cell")
