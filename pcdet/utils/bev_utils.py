from functools import lru_cache
from pathlib import Path
from numpy import float32, ndarray, array
from typing import Union, List, Tuple, DefaultDict
from numpy.typing import NDArray
from collections import defaultdict


# ==============================================================================
# LOAD KITTI DATA AS SINGLE FILE, SEVERAL FILES AND .PKL FILE
# ==============================================================================
@lru_cache(maxsize=None)
def load_kitti_pointcloud(bin_path: str = None, pkl_path: str = None, sample_idx: int = 0, augment_idx: int = 0) -> Union[ndarray, List[ndarray]]:
    from os import listdir
    from numpy import float32, fromfile, load

    if bin_path:
        bin_path = Path(bin_path)

        if bin_path.is_dir(): # several .bin files
            point_clouds = []
            bin_files = sorted([f for f in listdir(bin_path) if f.endswith('.bin')])

            for filename in bin_files:
                file_path = bin_path / filename
                if file_path.is_file():
                    try:
                        # N x 4 array (x, y, z, intensity)
                        point_cloud = fromfile(file_path, dtype=float32).reshape(-1, 4)
                        point_clouds.append(point_cloud)
                    except Exception as e:
                        print(f'Error during loading {filename}: {e}')
            return point_clouds
        
        elif bin_path.is_file(): # single .bin file
            return fromfile(bin_path, dtype=float32).reshape(-1, 4)
        else:
            raise FileNotFoundError(f'.bin file or directory not found: {bin_path}')
    
    elif pkl_path:
        pkl_path = Path(pkl_path)

        if not pkl_path.is_file():
            raise FileNotFoundError(f".pkl file not found: {pkl_path}")
        
        # Loading training entries (slow)
        #with open(pkl_path, 'rb') as f:
        #    data = load(f)

        # Loading training entries (faster)
        data = load(pkl_path, allow_pickle=True)
        num_samples = len(data)
            
        try:
            #if isinstance(data, list): # old
            # Access the sample at sample_idx
            sample = data[sample_idx]

            if 'frame_id' in str(sample): # train .pkl
                # Train data for BEV
                selected_data = sample[augment_idx]
                point_cloud = selected_data['points']
                curr_frame_id = selected_data['frame_id']
                # Labels for BEV train
                gt_boxes = selected_data['gt_boxes']
    
                return point_cloud, gt_boxes, num_samples, curr_frame_id
            
            elif 'point_cloud' in sample: # val .pkl
                # Val data for BEV
                sample = data[sample_idx]
                point_cloud = sample['points']
                lidar_idx = sample['point_cloud'].get('lidar_idx', 'unknown')
                # Labels for BEV val
                gt_boxes_lidar = sample['annos']['gt_boxes_lidar']
                truncated = sample['annos']['truncated']
                occluded = sample['annos']['occluded']
                box_2d = sample['annos']['bbox']

                # Pack additional information for validation
                val_info = {
                    'bbox': box_2d,
                    'truncated': truncated,
                    'occluded': occluded
                }

                return point_cloud, gt_boxes_lidar, val_info, num_samples, lidar_idx
            
        except IndexError:
            raise ValueError(f"Invalid sample_idx {sample_idx} or augment_idx {augment_idx} for .pkl dataset.")
    else:
        raise ValueError("Either bin_path or pkl_path must be provided.")

# ==============================================================================
# LOAD LAST PROCESSED SAMPLE
# ==============================================================================
@lru_cache(maxsize=None)
def load_progress(progress_file: str) -> Tuple[int, int, DefaultDict[str, int]]:

    if Path(progress_file).exists():
        from json import load

        with open(progress_file, "r") as f:
            progress = load(f)
            
        frame_id_dict = defaultdict(int)
        frame_id_dict.update(progress.get("processed_frames", {}))
        
        return (
            progress.get("sample_idx", 0),
            progress.get("augment_idx", 0),
            frame_id_dict
        )
    return 0, 0, defaultdict(int)   

# ==============================================================================
# SAVE RECENT PROCESSED SAMPLE
# ==============================================================================
def save_progress(sample_idx: int, augment_idx: int, frame_id_dict: DefaultDict[str, int], progress_file: str) -> None:
    from json import dump

    progress = {
        "sample_idx": sample_idx,
        "augment_idx": augment_idx,
        "processed_frames": dict(frame_id_dict)
    }
    with open(progress_file, "w") as f:
        dump(progress, f, indent=2)

# ==============================================================================
# CREATES UNIQUE FRAME ID
# ==============================================================================
def get_unique_frame_id(frame_id: str, augment_idx: int, frame_id_dict: DefaultDict[str, int]) -> Tuple[str, DefaultDict[str, int]]:
    needs_suffix = frame_id_dict[frame_id] > 0
    base_id = f"{frame_id}_r" if needs_suffix else frame_id
    
    if augment_idx == 4:
        frame_id_dict[frame_id] += 1
        
    return base_id, frame_id_dict

# ==============================================================================
# MAP HEIGHT (M) TO 255 (PX)
# ==============================================================================
@lru_cache(maxsize=None)
def map_height2channel(val_m: float, OFFSET_LIDAR: float, Z_MIN_HEIGHT: float, Z_MAX_HEIGHT: float, OUT_MIN: int, OUT_MAX: int) -> int:
    # Shift z-coordinate by LiDAR offset
    val_m += OFFSET_LIDAR
    # Normalize the height value to the range 0...1
    normalized_value = (val_m - Z_MIN_HEIGHT) / (Z_MAX_HEIGHT - Z_MIN_HEIGHT)
    # Alternative method to ensure the value stays within [0.0, 1.0]
    # This is commented out but can be used to limit the range of normalized values
    #normalized_value = min(1.0, max(0.0, normalized_value))
    # Scaling to the target range [OUT_MIN, OUT_MAX]
    scaled_value = normalized_value * (OUT_MAX - OUT_MIN) + OUT_MIN
    # see above
    #scaled_value = normalized_value * 255.0
    # Limitation of the value to the target range [OUT_MIN, OUT_MAX]
    clamped_value = min(OUT_MAX, max(OUT_MIN, scaled_value))
    # Rounding and conversion to int
    final_value = round(clamped_value)

    return int(final_value)

# ==============================================================================
# MAP INTENSITY (0...1) TO 255 (PX)
# ==============================================================================
@lru_cache(maxsize=None)
def map_intensity2channel(val_int: float, MAX_INTENSITY: float) -> int:
    # Scaling the intensity (use this when your val_int. values are 0...255)
    #scaled_intensity = (val_int * 255.0) / self.MAX_INTENSITY
    # Unmap 0->1 to 0->255 (use this when you val_int is alreasy in 0...1)
    scaled_intensity = val_int * 255.0
    # Limit the intensity to the range 0-255
    clamped_intensity = min(255, scaled_intensity) if scaled_intensity > 0 else 0

    return int(clamped_intensity)

# ==============================================================================
# MAP DENSITY TO 255 (PX)
# ==============================================================================
@lru_cache(maxsize=None)
def map_density2channel(val_den: int, MAX_DENSITY: float) -> int:
    # Scaling the density
    scaled_density = (val_den * 255.0) / MAX_DENSITY
    # Limit the density to the range 0-255
    clamped_density = min(255, scaled_density) if scaled_density > 0 else 0

    return int(clamped_density)

# ==============================================================================
# CONVERT POINT CLOUD (COORDS.) TO IMAGE ROW/COLUMN (COORDS.)
# ==============================================================================
def map_pc2rc(x: float, y: float, row: int, column: int, IMAGE_HEIGHT: int, IMAGE_WIDTH: int, CELL_SIZE: float) -> int:
    # 3D point cloud value -> row 2D Mapping
    row[0] = int(round(((IMAGE_HEIGHT * CELL_SIZE) / 1.0 - x) / (IMAGE_HEIGHT * CELL_SIZE) * IMAGE_HEIGHT))

    # 3D point cloud value -> column 2D Mapping
    column[0] = int(round(((IMAGE_WIDTH * CELL_SIZE) / 2.0 - y) / (IMAGE_WIDTH * CELL_SIZE) * IMAGE_WIDTH))
   
    return 1 # Return success,if in range and row/column are set

# ==============================================================================
# CONVERT IMAGE ROW/COLUMN (COORDS.) TO POINT CLOUD (COORDS.)
# ==============================================================================
def map_rc2pc(x: float, y: float, row: int, column: int, IMAGE_HEIGHT: int, IMAGE_WIDTH: int, CELL_SIZE: float) -> int:
    if 0 <= row < IMAGE_HEIGHT and 0 <= column < IMAGE_WIDTH:
        # row pixel value -> x Mapping (2D)
        x[0] = float(CELL_SIZE *-1.0 * (row - (IMAGE_HEIGHT / 1.0))) # coordinate system: 1.0 bottom center, 2.0 middle center

        # column pixel value -> y Mapping (2D)
        y[0] = float(CELL_SIZE *-1.0 * (column - (IMAGE_WIDTH / 2.0)))
      
        return 1 # Return success, if in range and x/y are set

    return 0

# ==============================================================================
# NORMALIZE BEV BOUNDING BOX COORDINATES [0...1]
# ==============================================================================
@lru_cache(maxsize=None)
def normalize_coordinates(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float, img_width: int, img_height: int
                          ) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    x_n = x / image_width
    y_n = y / image_height
    """
    x1 /= img_width
    x2 /= img_width
    x3 /= img_width
    x4 /= img_width
    y1 /= img_height
    y2 /= img_height
    y3 /= img_height
    y4 /= img_height

    return x1, y1, x2, y2, x3, y3, x4, y4

# ==============================================================================
# COMPUTES THE DENSITY DISTRIBUTION AND MAX DENSITY
# ==============================================================================
def max_density_distribution(density_array: NDArray[float32], image_height: int, image_width: int, distribution_bins: int = 10) -> None:
    max_density = 0
    distribution = [0] * distribution_bins  # Predefine distribution array with `distribution_bins` elements

    for i in range(image_height):
        for j in range(image_width):
            # Find max density
            if density_array[i][j] > max_density:
                max_density = density_array[i][j]

            # Find distribution of points
            density_value = density_array[i][j]
            
            # Make sure the distribution array can accommodate the density value
            if density_value >= len(distribution):
                distribution.extend([0] * (density_value - len(distribution) + 1))

            distribution[int(density_array[i][j])] += 1

    print(f"Max density: {max_density}")

    print("Max Distribution:")
    for i in range(min(distribution_bins, len(distribution))):
        print(f"Cells with {i} points: {distribution[i]}")

# max_density = 0
#         distribution = [0] * 10 # higher values had no influence
#         for i in range(self.config.IMAGE_HEIGHT):
#             for j in range(self.config.IMAGE_WIDTH):

#                 # find max density
#                 if self.densityArray[i][j] > max_density:
#                     max_density = self.densityArray[i][j]

#                 # find distribution of points
#                 # Note: this prevents list index put of range mode='pkl'
#                 density_value = self.densityArray[i][j]
#                 if density_value >= len(distribution):
#                     distribution.extend([0] * (density_value - len(distribution) + 1))

#                 distribution[int(self.densityArray[i][j])] += 1
            
#             print(f"Max density: {max_density}")

#             print("Max Distribution:")
#             for i in range(10):
#                 print(f"Cells with {i} points: {distribution[i]}")