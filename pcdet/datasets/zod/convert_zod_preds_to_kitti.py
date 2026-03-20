"""Converts YOLO11 OBB model predictions to KITTI format.
   Compatible with ZOD SDK.

   Author: Petros Katsoulakos, 20 February 2026
"""
import argparse
import os
import pickle
from os.path import join
from typing import List, Dict, Any

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from zod.constants import Camera, Lidar
from zod.data_classes.box import Box2D, Box3D
from zod.data_classes.calibration import Calibration, LidarCalibration, CameraCalibration
from zod.data_classes.geometry import Pose
from pcdet.utils import common_utils

# Convert KITTI (x-forward, y-left, z-up) to ZOD (y-forward, x-right, z-up)
Tr_KITTI_LiDAR_to_ZOD_LiDAR = np.array([[0, -1, 0], # x_zod = -y_kitti
                                        [1, 0, 0], # y_zod = x_kitti   
                                        [0, 0, 1]]) # z_zod = z_kitti

def build_calibration_from_pkl(calib_dict: Dict[str, Any]) -> Calibration:
    lidar_extrinsics = np.array(calib_dict['lidar_extrinsics'])
    cam_extrinsics = np.array(calib_dict['cam_extrinsics'])
    cam_intrinsics = np.array(calib_dict['cam_intrinsics'])
    distortion = np.array(calib_dict['cam_distortion'])
    undistortion = np.array(calib_dict['cam_undistortion'])
    field_of_view = np.array(calib_dict['cam_field_of_view'])
    image_shape = np.array(calib_dict['image_shape'][::-1]) # width, height

    lidars = {Lidar.VELODYNE: LidarCalibration(extrinsics=Pose(lidar_extrinsics))}
    cameras = {Camera.FRONT: CameraCalibration(extrinsics=Pose(cam_extrinsics),
                                               intrinsics=cam_intrinsics,
                                               distortion=distortion,
                                               undistortion=undistortion,
                                               image_dimensions=image_shape,
                                               field_of_view=field_of_view)}
    
    return Calibration(lidars=lidars, cameras=cameras)

def pixel_to_world_coordinates(center_px: tuple[float, float], dim_px: tuple[float, float], image_width: int = 1024, image_height: int = 1024, cell_size: float = 0.06) -> tuple[tuple[float, float], tuple[float, float]]:
    """Convert YOLO predictions format from pixel to world coordinates (ZOD LiDAR frame)."""
    # only for x-forward, y-left coordinate system (KITTI)
    y = -(center_px[0] - image_width / 2) * cell_size # Image x -> -LiDAR y
    x = (image_height - center_px[1]) * cell_size # Image y -> LiDAR x
    # align to y-forward, x-right coordinate system (ZOD)
    xyz_kitti = np.array([x, y, 0.0]) 
    xyz_zod = xyz_kitti @ Tr_KITTI_LiDAR_to_ZOD_LiDAR.T
    x_zod, y_zod, _ = xyz_zod
    l = dim_px[0] * cell_size # width is horizontal in OpenCV = length in ZOD
    w = dim_px[1] * cell_size # height is vertical in OpenCV = width in ZOD
    
    return (x_zod, y_zod), (l, w)

def bev_preds_to_ZOD_lidar(boxes_bev: list[list[float]], image_width: int = 1024, image_height: int = 1024, cell_size: float = 0.06) -> np.ndarray:
    """
    Transform YOLO BEV predictions (N, 7) [cls_id, cx, cy, w, h, r, conf]
    to ZOD-LiDAR-Frame: [class_id, x, y, z, l, w, h, heading, conf]
    """
    boxes_lidar = []

    for box in boxes_bev:
        cls_id, cx, cy, w, h, r, conf = box
        (x, y), (l, w) = pixel_to_world_coordinates((cx, cy), (w, h), image_width=image_width, 
                                                        image_height=image_height, cell_size=cell_size)
        heading = r #+ np.pi/2 # adjust heading from BEV to LiDAR (TODO: -r + np.pi/2)
        heading = common_utils.limit_period_numpy(heading, offset=0.5, period=2 * np.pi)
        default_heights = {1: 1.54, 2: 1.67, 3: 1.75}  # Car: 1.54m, Ped: 1.67m, Cyc: 1.75m 
        height = default_heights.get(cls_id)
        z = -1.59 # approx. value fitting for 3D Boxes vs 3D GT # 1.55 or 1.59
        class_map = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        obj_type = class_map.get(cls_id, "DontCare")
        boxes_lidar.append([obj_type, x, y, z, l, w, height, heading, conf])
    
    return np.array(boxes_lidar)

def _convert_to_kitti(annos: Dict[str, Any]) -> List[str]:
    """
    Convert predictions labels to KITTI format lines.
    Expects annos dict with keys:
      - name: List[str]
      - truncated: List[float]
      - occluded: List[int]
      - bbox: List[List[float]]
      - gt_boxes_camera: np.ndarray (N, 7) [x, y, z, h, w, l, rotation_y]
      - score: np.ndarray (N,)
    """
    kitti_annotation_lines = []
    num_objs = len(annos['name'])
    for i in range(num_objs):
        name = annos['name'][i]
        truncated = annos['truncated'][i]
        occluded = annos['occluded'][i]
        bbox = annos['bbox'][i] # xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        box3d = annos['pred_boxes_camera'][i]
        size = box3d[3:6] # h, w, l (Camera format)
        location = box3d[:3] # x, y, z
        rotation_y = box3d[6]   # rotation_y (Camera format)
        # Calculate alpha 
        # alpha = r_y - theta; theta = arctan2(x, z)
        alpha = rotation_y - np.arctan2(location[0], location[2]) # x, z (Camera coordinates)
        score = annos['score'][i]

        kitti_obj = " ".join([
            str(name),
            f"{float(truncated):.2f}",
            f"{int(occluded)}",
            f"{float(alpha):.2f}",
            f"{xmin:.2f}", f"{ymin:.2f}", f"{xmax:.2f}", f"{ymax:.2f}",
            f"{size[0]:.2f}", f"{size[1]:.2f}", f"{size[2]:.2f}",
            f"{location[0]:.2f}", f"{location[1]:.2f}", f"{location[2]:.2f}",
            f"{rotation_y:.2f}",f"{score:.3f}"
            ])
        kitti_annotation_lines.append(kitti_obj)
    return kitti_annotation_lines

def _lidar_to_camera(boxes_lidar: np.ndarray, calib_dict: Dict[str, Any], class_names: list[str]) -> np.ndarray:
    """
    Transforms boxes from the LiDAR frame to the camera frame.
    Expected: boxes_lidar in the ZOD LiDAR frame, calib_dict from .pkl.
    Returns: Array of [x, y, z, h, w, l, yaw] in camera coordinates
    """
    calib = build_calibration_from_pkl(calib_dict)
    boxes_camera = []

    # TODO: prüfen, ob vor oder nach Transformation
    boxes_lidar = boxes_lidar.copy()
    boxes_lidar[:, 2] -= boxes_lidar[:, 5] / 2 # shift box center LiDAR to Camera (KITTI definition)

    for box in boxes_lidar:
        size = box[3:6]
        location = box[:3] 
        yaw = box[6] 
        
        #yaw = common_utils.limit_period_numpy(yaw, offset=0.5, period=2 * np.pi)
        # test if necessary
        orientation = Quaternion(axis=[0, 0, 1], radians=yaw)
        box3d = Box3D(center=location, size=size, orientation=orientation, frame=Lidar.VELODYNE)
        # Transform the 3D LiDAR Box to 3D Camera Box
        box3d.convert_to(Camera.FRONT, calib)
        box3d_cam = np.zeros(7)
        box3d_cam[:3] = box3d.center
        box3d_cam[3:6] = box3d.size[::-1] # l, w, h -> h, w, l
        box3d_cam[6] = box3d.orientation.yaw_pitch_roll[0]
        #box3d_cam[1] -= box3d_cam[3] / 2 # y = -h/2
        boxes_camera.append(box3d_cam)
    
    return np.array(boxes_camera)

def convert_prediction(
    frame_id: str,
    preds_lidar: np.ndarray,
    calib: dict,
    image_shape: np.ndarray,
    target_path: str,
) -> None:
    """Convert a single prediction from .txt to KITTI format."""
    class_names = preds_lidar[:, 0].tolist()
    pred_boxes_lidar = preds_lidar[:, 1:8].astype(float) # x, y, z, l, w, h, heading
    scores = preds_lidar[:, 8].astype(float)# confidence

    calib = calib.copy()
    calib['image_shape'] = image_shape

    # Transform ZOD LiDAR boxes to ZOD Camera Boxes
    boxes_zod_camera = _lidar_to_camera(pred_boxes_lidar, calib, class_names=class_names)

    annos = {
        'name': class_names,
        'truncated': [-1.0] * len(class_names),
        'occluded': [-1] * len(class_names),
        'bbox': [[-1, -1, -1, -1]] * len(class_names),
        'pred_boxes_camera': boxes_zod_camera,
        'score': scores
    }

    kitti_lines = _convert_to_kitti(annos)

    if len(annos['name']) == 0:
        return

    # Write output file
    output_filename = f"{frame_id}.txt"
    with open(join(target_path, output_filename), "w") as target_file:
        target_file.write("\n".join(kitti_lines))

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert ZOD .txt predictions to KITTI format")
    parser.add_argument("--pkl-path", required=True, help="Path to zod_val_dataset.pkl")
    parser.add_argument("--target-dir", required=True, help="Output directory for KITTI labels")
    parser.add_argument("--preds-dir", required=True, help="Directory containing YOLO predictions (.txt files)")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    print(f"WARNING: CONVERTING FROM BEV IMAGE SPACE over ZENSEACT LIDAR COORDINATE FRAME TO KITTI CAMERA COORDINATE FRAME")
    assert args.target_dir not in args.pkl_path, "Do not write to the dataset"
    
    print("Loading ZOD calibration data...")
    with open(args.pkl_path, "rb") as f:
        dataset = pickle.load(f)
    calib_dict = {frame['point_cloud']['lidar_idx']: frame['calib'] for frame in dataset}
    image_shape_dict = {frame['point_cloud']['lidar_idx']: frame['image']['image_shape'] for frame in dataset}

    pred_files = sorted([f for f in os.listdir(args.preds_dir) if f.endswith('.txt')])
    
    # Use this, if you want to test one specific sample
    #pred_files = ['000135.txt']

    # Create target directory
    os.makedirs(args.target_dir, exist_ok=True)  
    
    # Convert all predictions
    for pred_file in tqdm(pred_files, desc="Converting predictions..."):
        frame_id = os.path.splitext(pred_file)[0]
        preds_path = os.path.join(args.preds_dir, pred_file)
        with open(preds_path, "r") as f:
            preds_lines = f.readlines()
        preds_bev = [list(map(float, line.strip().split())) for line in preds_lines]
        preds_lidar = bev_preds_to_ZOD_lidar(preds_bev)
        calib = calib_dict[frame_id]
        image_shape = image_shape_dict[frame_id]

        try:
            convert_prediction(
                frame_id,
                preds_lidar,
                calib,
                image_shape,
                args.target_dir,
            )
        except Exception as err:
            print(f"Failed converting prediction: {frame_id} with error: {str(err)}")
            raise
    
    print(f"Conversion complete. Output written to {args.target_dir}")


if __name__ == "__main__":
    main()

# TODO: wieso sind die mAP niedriger als KITTI?
# TODO: Skripte beide vereinheitlichen
# TODO: überlegen wie Level verbessert werden kann