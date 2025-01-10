import numpy as np
import glob
from progress.bar import IncrementalBar

def load_kitti_bin(bin_path: str) -> np.ndarray:
    """
    Loads a KITTI .bin Point Cloud and returns only x, y, z coordinates.
    """
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # [x, y, z]

def compute_density(point_cloud: np.ndarray, cell_size: float = 0.1) -> np.ndarray:
    """
    Computes the point density (points/0.1m²) over a dataset of KITTI point clouds.
    
    Parameters:
    - point_cloud: Point cloud data (numpy array of shape [n, 3], with x, y, z coordinates).
    - cell_size: Size of each small grid cell (e.g., 0.1 m).
    
    Returns:
    - density_map: 2D array of density values (points per 0.1m²) for each region.
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

    # Clip indices to ensure they are within valid bounds 
    # (0 <= x_indices < grid_x) and (0 <= y_indices < grid_y)
    x_indices = np.clip(x_indices, 0, grid_x - 1)
    y_indices = np.clip(y_indices, 0, grid_y - 1)

    density_map = np.zeros((grid_y, grid_x), dtype=np.int32)
    np.add.at(density_map, (y_indices, x_indices), 1)

    total_cells = grid_x * grid_y
    # x_indices = np.clip(x_indices, 0, grid_x - 1) # Make sure that x_indices remain within the grid limits
    # y_indices = np.clip(y_indices, 0, grid_y - 1) # Make sure that y_indices remain within the grid limits
    
    # cells = np.zeros((grid_y, grid_x), dtype=np.int32)
    # # Count points in each cell
    # np.add.at(cells, (y_indices, x_indices), 1)

    # # Sum points into 1m² regions
    # region_x = grid_x // region_size
    # region_y = grid_y // region_size

    # density_map = cells[:region_y * region_size, :region_x * region_size].reshape(
    #     region_y, region_size, region_x, region_size
    # ).sum(axis=(1, 3))

    # Calculate density in points per m²
    # density_map = density_map / (region_size * cell_size) ** 2
   
    return density_map, total_cells

# def process_bin_files(folder_path: str) -> tuple:
#     """
#     Processes all .bin files in the specified folder and computes density maps.
    
#     Returns:
#     - densities: List of density maps for each file.
#     - max_density: Maximum density across all files.
#     - avg_density: Average density across all files.
#     """
#     # List all .bin files in the folder
#     bin_files = glob.glob(f"{folder_path}/*.bin")
#     file_count = len(bin_files)

#     densities, all_densities = [], [] # To track densities for computing max and avg

#     # Loop through all .bin files
#     with IncrementalBar('Processing files', max=file_count, suffix='%(percent).1f%% - Estimated time: %(eta)ds') as bar:
#         for i, bin_file in enumerate(bin_files):
#             # Load point cloud from .bin file
#             point_cloud = load_kitti_bin(bin_file)
            
#             # Compute the density map for the current file
#             density_map, t_cells = compute_density(point_cloud)
#             densities.append(density_map)
            
#             # Flatten density_map and append to all_densities for max and avg calculation
#             #all_densities.append(density_map)

#             # Update progress bar
#             bar.next()

#     bar.finish()

#     # Calculate the max and avg densities
#     all_densities_flat = np.concatenate([dm.flatten() for dm in densities])
#     max_density = np.max(all_densities_flat)
#     avg_density = np.mean(all_densities_flat)

#    return densities, max_density, avg_density, t_cells

def process_bin_files(folder_path: str, file_path: str) -> tuple:
    bin_files = glob.glob(f"{folder_path}/*.bin")
    file_count = len(bin_files)
    #file_count = len(file_path)

    max_density = 0
    total_density_sum = 0
    total_points = 0

    with IncrementalBar('Processing files', max=file_count, suffix='%(percent).1f%% - Estimated time: %(eta)ds') as bar:
        for bin_file in bin_files:
            point_cloud = load_kitti_bin(bin_file)
            #point_cloud = load_kitti_bin(file_path)
            density_map, t_cells = compute_density(point_cloud)
            
            # Update max density
            max_density = max(max_density, np.max(density_map))

            # Update total sum for average calculation
            total_density_sum += np.sum(density_map)
            total_points += density_map.size

            # Clear variables to free memory
            del point_cloud, density_map

            bar.next()
    bar.finish()

    # Calculate average density
    avg_density = total_density_sum / total_points

    return max_density, avg_density, t_cells


if __name__ == "__main__":
    kitti_folder = "/home/rlab10/Downloads/training/velodyne"
    max_density, avg_density, t_cells = process_bin_files(kitti_folder, None)

    #bin_file = "/home/rlab10/Downloads/training/velodyne/000000.bin"
    #max_density, avg_density, t_cells = process_bin_files(None, bin_file)

    print(f'Total grid cells: {t_cells}')
    print(f"Maximum Density: {max_density:.2f} Points/0.1m² cell")
    print(f"Average Density: {avg_density:.2f} Points/0.1m² cell")