import os
import numpy as np
import yaml
import math
from numba import cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@cuda.jit
def transform_points_to_voxels(points, voxel_size, point_cloud_range, voxel_grid):
    # Calculate the Thread-ID
    idx = cuda.grid(1)  
    if idx < points.shape[0]:  # make sure, that the index between the borders
        point = points[idx]

        # Calculate the Voxel-Indices
        voxel_idx_x = math.floor(((point[0] - point_cloud_range[0]) / voxel_size[0]))
        voxel_idx_y = math.floor(((point[1] - point_cloud_range[1]) / voxel_size[1]))
        voxel_idx_z = math.floor(((point[2] - point_cloud_range[2]) / voxel_size[2]))


        # Check, if the point is in the range
        if (point[0] >= point_cloud_range[0] and point[0] <= point_cloud_range[3] and
            point[1] >= point_cloud_range[1] and point[1] <= point_cloud_range[4] and
            point[2] >= point_cloud_range[2] and point[2] <= point_cloud_range[5]):
            
            # Check the indices and increase the counter in the voxel grid
            if (0 <= voxel_idx_x < voxel_grid.shape[0] and
                0 <= voxel_idx_y < voxel_grid.shape[1] and
                0 <= voxel_idx_z < voxel_grid.shape[2]):
                # Atomic addition to increase the counter
                cuda.atomic.add(voxel_grid, (voxel_idx_x, voxel_idx_y, voxel_idx_z), 1)


def load_bin_file(file_path):
    try:
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def sort_key(f):
    return int(f[:-4])

def process_all_bin_files(directory, voxel_size, point_cloud_range):
    max_voxel_count = 0
    bin_files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    bin_files.sort(key=sort_key)

    print(f"Found {len(bin_files)} .bin files.")
    
    for bin_file in bin_files:
        file_path = os.path.join(directory, bin_file)
        print(f"Processing file: {file_path}")

        # Load the points from CPU memory
        points = load_bin_file(file_path)

        # Transfer the points to the GPU
        d_points = cuda.to_device(points)

        # Define block and grid sizes for CUDA
        blockdim = 256  # Number of threads per block
        griddim = (d_points.shape[0] + (blockdim - 1)) // blockdim  # Number of blocks
        #griddim = max((d_points.shape[0] + (blockdim - 1)) // blockdim, 48)

        voxel_size = voxel_size.astype(np.float32)
        point_cloud_range = point_cloud_range.astype(np.float32)

        # Create the voxel grid on the CPU and then transfer it to the GPU
        voxel_grid_shape = (
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]) + 1,
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]) + 1,
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]) + 1)

        voxel_grid = np.zeros(voxel_grid_shape, dtype=np.int32)
        d_voxel_grid = cuda.to_device(voxel_grid)

        # Call the CUDA kernel to transform the points into voxels
        transform_points_to_voxels[griddim, blockdim](d_points, voxel_size, point_cloud_range, d_voxel_grid)

        # Copy the voxel grid back to the CPU
        voxel_grid = d_voxel_grid.copy_to_host()

        # Count the voxels
        voxel_count = np.sum(voxel_grid > 0)  # Count the voxels that contain points
        print(f"Voxel count for {bin_file}: {voxel_count}")

        if voxel_count > max_voxel_count:
            max_voxel_count = voxel_count

    return max_voxel_count

def main():
    directory = '/home/rlab10/OpenPCDet/data/kitti/training/velodyne/'
    yaml_file = '/home/rlab10/OpenPCDet/tools/cfgs/dataset_configs/custom_dataset.yaml'
    
    config = load_config(yaml_file)
    voxel_size = np.array(config['DATA_PROCESSOR'][2]['VOXEL_SIZE'], dtype=np.float32)
    point_cloud_range = np.array(config['POINT_CLOUD_RANGE'], dtype=np.float32)

    print(f"Voxel Size: {voxel_size}")
    print(f"Point Cloud Range: {point_cloud_range}") 

    max_voxel_count = process_all_bin_files(directory, voxel_size, point_cloud_range)

    print(f"The maximum number of voxels across all .bin files is: {max_voxel_count}")

if __name__ == "__main__":
    main()



