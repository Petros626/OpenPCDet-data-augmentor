import numpy as np
import glob
from tqdm import tqdm

def analyze_kitti_data(folder_path, voxel_size=0.1, point_cloud_range=[0, -30, -2.73, 60, 30, 1.27]): 
    """
    Analyzes maximum intensity, density and density distribution from KITTI data set
    """
    max_intensity = 0
    max_density = 0
    density_distribution = [0] * 10
    total_points = 0

    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range # pc_range was from kitti_dataset.yaml, change to your values
    N_x = int((x_max - x_min) / voxel_size)
    N_y = int((y_max - y_min) / voxel_size)
    N_z = int((z_max - z_min) / voxel_size)

    total_voxels = N_x * N_y * N_z
    print(f"Total possible Voxel Count {total_voxels} (based on voxel size of {voxel_size}m)")

    bin_files = glob.glob(f"{folder_path}/*.bin")
    total_files = len(bin_files)

    print(f'Analyze {total_files} files...')

    for bin_file in tqdm(bin_files, desc='Progress', unit='file'):

        point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        intensities = point_cloud[:, 3] # [x, y, z, intensity]

        total_points += point_cloud.shape[0] # [x, y, z]
        
        max_intensity = max(max_intensity, np.max(intensities))
        
        voxel_indices = ((point_cloud[:, :2] - [x_min, y_min]) / voxel_size).astype(int)

        _, counts = np.unique(voxel_indices, axis=0, return_counts=True)

        for count in counts:
            max_density = max(max_density, count) 
            if count < len(density_distribution):  
                density_distribution[count] += 1
            else:
                while count >= len(density_distribution):
                    density_distribution.append(0)
                density_distribution[count] += 1

    avg_density = total_points / total_voxels if total_voxels > 0 else 0
    
    print(f"Total Points in KITTI .bin files: {total_points}")

    return max_intensity, max_density, avg_density, density_distribution


kitti_folder = "/home/rlab10/Downloads/training/velodyne" 
max_intensity, max_density, avg_density, density_distribution = analyze_kitti_data(kitti_folder)

print(f"Maximum Intensity: {max_intensity}")
#print(f"Maximum Density: {max_density}")
#print(f"Average Density: {avg_density:.4f}")


