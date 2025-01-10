import os
import numpy as np
import yaml
from numba import jit,cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@jit(target_backend='cuda', forceobj=True)	
def transform_points_to_voxels(points, voxel_size, point_cloud_range):
    grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)

    # dict
    voxel_grid = {}
    for point in points:
        # filter areas outside cloud
        if np.any(point[:3] < point_cloud_range[:3]) or np.any(point[:3] >= point_cloud_range[3:6]):
            continue
        # calculate index for each point
        voxel_idx = np.floor((point[:3] - point_cloud_range[:3]) / voxel_size).astype(int)
        voxel_idx = tuple(voxel_idx)
        # index not exist, create new
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = []
        
        voxel_grid[voxel_idx].append(point)

    return voxel_grid

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

@jit(target_backend='cuda', forceobj=True)	
def process_all_bin_files(directory, voxel_size, point_cloud_range):
    max_voxel_count = 0
    bin_files = [f for f in os.listdir(directory) if f.endswith('.bin')]
    bin_files.sort(key=sort_key)

    print(f"Found {len(bin_files)} .bin files.")
    
    for bin_file in bin_files:
        file_path = os.path.join(directory, bin_file)
        print(f"Processing file: {file_path}")
        points = load_bin_file(file_path)

        voxel_grid = transform_points_to_voxels(points, voxel_size, point_cloud_range)
        voxel_count  = len(voxel_grid)
        print(f"Voxel count for {bin_file}: {voxel_count}")

        if voxel_count > max_voxel_count:
            max_voxel_count = voxel_count

    return max_voxel_count

def main():
    directory = '/home/rlab10/OpenPCDet/data/kitti/training/velodyne/'
    yaml_file = '/home/rlab10/OpenPCDet/tools/cfgs/dataset_configs/custom_dataset.yaml'
    
    config = load_config(yaml_file)
    voxel_size = np.array(config['DATA_PROCESSOR'][2]['VOXEL_SIZE'])
    point_cloud_range = np.array(config['POINT_CLOUD_RANGE'])

    print(f"Voxel Size: {voxel_size}")
    print(f"Point Cloud Range: {point_cloud_range}") 

    max_voxel_count = process_all_bin_files(directory, voxel_size, point_cloud_range)

    print(f"The maximum number of voxels across all .bin files is: {max_voxel_count}")

if __name__ == "__main__":
    main()



