"""Converts ZOD dynamic object annotations to KITTI format.
   Compatible with ZOD SDK.

   Original author: Roland Meertens, 10 February 2022
   Original source: https://github.com/zenseact/EdgeAnnotationZChallenge/pull/1/changes

   This is a lightly modified version by Petros Katsoulakos, 17 February 2026, 
   adapted for own purpose.
"""
# source: https://github.com/zenseact/EdgeAnnotationZChallenge/pull/1/changes

import argparse
import os
import pickle
from os.path import join
from typing import List, Dict, Any

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from zod.constants import Camera, Lidar
from zod.data_classes.box import Box3D
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

def kitti_lidar_back_to_zod_lidar(boxes_lidar: np.ndarray) -> np.ndarray:
    """Transform boxes from KITTI LiDAR to ZOD LiDAR frame."""
    boxes = boxes_lidar.copy()
    # Transform center
    boxes[:, :3] = boxes[:, :3] @ Tr_KITTI_LiDAR_to_ZOD_LiDAR.T # x, y, z
    # Adjust yaw: yaw_zod = yaw_kitti + pi/2
    boxes[:, 6] = boxes [:, 6] + np.pi/2 # yaw
    # Normalize yaw to [-pi, pi]
    boxes[:, 6] = common_utils.limit_period(boxes[:, 6], offset=0.5, period=2 * np.pi)

    return boxes

def get_offset_for_classes(class_name):
    if class_name == "Car":
        return 0.4
    elif class_name in {"Pedestrian", "Cyclist"}:
        return 0.3
    else:
        return 0.0

def _convert_to_kitti(annos: Dict[str, Any]) -> List[str]:
    """
    Convert .pkl annotation dict to KITTI format lines.
    Expects annos['gt_boxes_camera'] in [x, y, z, h, w, l, yaw] per box.
    """
    kitti_annotation_lines = []
    num_objs = len(annos['name'])
    for i in range(num_objs):
        name = annos['name'][i]
        truncated = annos['truncated'][i]
        occluded = annos['occluded'][i]
        bbox = annos['bbox'][i] # xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        box3d = annos['gt_boxes_camera'][i]
        size = box3d[3:6] # h, w, l (Camera format)
        location = box3d[:3] # x, y, z
        rotation_y = box3d[6]   # rotation_y (Camera format)
        # Calculate alpha 
        # alpha = r_y - theta; theta = arctan2(x, z)
        alpha = rotation_y - np.arctan2(location[0], location[2]) # x, z (Camera coordinates)

        kitti_obj = " ".join([
            str(name),
            f"{float(truncated):.2f}",
            f"{int(occluded)}",
            f"{float(alpha):.2f}",
            f"{xmin:.2f}", f"{ymin:.2f}", f"{xmax:.2f}", f"{ymax:.2f}",
            f"{size[0]:.2f}", f"{size[1]:.2f}", f"{size[2]:.2f}",
            f"{location[0]:.2f}", f"{location[1]:.2f}", f"{location[2]:.2f}",
            f"{rotation_y:.2f}"
            ])
        kitti_annotation_lines.append(kitti_obj)
    return kitti_annotation_lines

def _lidar_to_camera(boxes_lidar: np.ndarray, calib_dict: Dict[str, Any], class_names: str, anno_offset: bool) -> np.ndarray:
    """
    Transforms boxes from the LiDAR frame to the camera frame.
    Expected: boxes_lidar in the ZOD LiDAR frame, calib_dict from .pkl.
    Returns: Array of [x, y, z, h, w, l, yaw] in camera coordinates
    """
    calib = build_calibration_from_pkl(calib_dict)
    boxes_camera = []

    boxes_lidar = boxes_lidar.copy()
    boxes_lidar[:, 2] -= boxes_lidar[:, 5] / 2 # shift box center LiDAR to Camera (KITTI definition)

    for i, box in enumerate(boxes_lidar):
        size = box[3:6]
        if anno_offset and class_names is not None:
            offset = get_offset_for_classes(class_names[i])
            size[0] += offset # length
            size[1] += offset # width
        location = box[:3] 
        yaw = box[6]
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

def convert_annotation(
    frame_data: Dict[str, Any],
    target_path: str,
    anno_offset: bool = False
) -> None:
    """Convert a single frame's annotations from .pkl to KITTI format."""
    frame_id = frame_data['point_cloud']['lidar_idx']
    image_shape = frame_data['image']['image_shape']
    annos = frame_data['annos']
    class_names = annos['name']
    calib = frame_data['calib']
    calib['image_shape'] = image_shape
    gt_boxes_lidar = annos['gt_boxes_lidar'] # x, y, z, l, w, h, heading
    # Transform the gt_boxes_lidar from KITTI to ZOD coordinate system
    boxes_zod_lidar = kitti_lidar_back_to_zod_lidar(gt_boxes_lidar)
    # Transform ZOD LiDAR boxes to ZOD Camera Boxes
    boxes_zod_camera = _lidar_to_camera(boxes_zod_lidar, calib, class_names=class_names, anno_offset=anno_offset)

    annos = annos.copy()
    annos['gt_boxes_camera'] = boxes_zod_camera

    kitti_lines = _convert_to_kitti(annos)

    if len(annos['name']) == 0:
        return

    # Write output file
    output_filename = f"{frame_id}.txt"
    with open(join(target_path, output_filename), "w") as target_file:
        target_file.write("\n".join(kitti_lines))

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert ZOD .pkl annotations to KITTI format")
    parser.add_argument("--pkl-path", required=True, help="Path to zod_val_dataset.pkl")
    parser.add_argument("--target-dir", required=True, help="Output directory for KITTI labels")
    parser.add_argument("--anno-offset", action="store_true", help="Training annos used offsets (Car: (l, w) + 0.4, Ped/Cyc: (l, w) + 0.3). Gt annos should also to avoid reduced mAP")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    print(f"WARNING: CONVERTING FROM ZENSEACT LIDAR COORDINATE FRAME TO KITTI CAMERA COORDINATE FRAME")
    assert args.target_dir not in args.pkl_path, "Do not write to the dataset"
    
    print("Loading ZOD data...")
    with open(args.pkl_path, "rb") as f:
        dataset = pickle.load(f)
    
    # Use this, if you want to test one specific sample
    #dataset = [dataset[3582]] 

    # Create target directory
    os.makedirs(args.target_dir, exist_ok=True)  
    
    # Convert all annotations
    for frame_data in tqdm(dataset, desc="Converting annotations..."):
        try:
            convert_annotation(
                frame_data, 
                args.target_dir,
                args.anno_offset
            )
        except Exception as err:
            print(f"Failed converting annotation: {frame_data['point_cloud']['lidar_idx']} with error: {str(err)}")
            raise
    
    print(f"Conversion complete. Output written to {args.target_dir}")


if __name__ == "__main__":
    main()