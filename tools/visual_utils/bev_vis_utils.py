"""
OpenCV BEV visualization tool box
Written by Petros626
All rights preserved from 2024 - present.
"""

from numpy import array, dot, count_nonzero, sum, uint8, float32
import numpy as np
from cv2 import line, arrowedLine, circle, getTextSize, putText, FONT_HERSHEY_DUPLEX, LINE_AA
from typing import List, Tuple, Union, Optional
from numpy.typing import NDArray


box_colormap = [
    [0, 0, 0], # 0- not assigned
    [0, 255, 0], # 1- Car Green
    [255, 0, 255], # 2 - Pedestrian Violet
    [0, 255, 255], # 3 - Cyclist Yellow
]

class_names = { 
    1: 'Car', 
    2: 'Ped',
    3: 'Cyc'
}

# ==============================================================================
# DRAWS A THICKER LINE IN OBJECT DIRECTION (HEADING)
# ==============================================================================
def draw_bbox_direction(bev_img: NDArray[uint8], bev_corners: NDArray[float32], color: Tuple[int, int, int] = (255, 255, 0), thickness: int = 2) -> None:
    corners_int = bev_corners.reshape(-1, 2).astype(int)

    # Top-right (x1, y1) to Bottom-right (x2, y2) to Bottom-left (x3, y3) to Top-left (x4, y4)
    line(bev_img, (corners_int[3, 0], corners_int[3, 1]), (corners_int[0, 0], corners_int[0, 1]), color, thickness)

# ==============================================================================
# DRAWS AN ARROW FROM BEV-BOX CENTER
# ==============================================================================
def draw_bbox_arrow(bev_img: NDArray[uint8], centroid: float, yaw: float, cls: int, bev_res: float, color: Tuple[int, int, int]) -> None:
    # Set arrow length based on class
    if cls == 1: # Car
        arrow_length = 3.5
    elif cls == 2 or cls == 3: # Pedestrian,  Cyclist
        arrow_length = 2.5
    # Calculate the arrow's end position using the centroid, yaw, and arrow length
    arr_end = centroid + array([arrow_length * np.cos(yaw), arrow_length * np.sin(yaw)])

    # Convert centroid and arrow end to image coordinates (BEV perspective)
    start_bevx = int(bev_img.shape[1] / 2 + (-centroid[1]) / bev_res)
    end_bevx = int(bev_img.shape[1] / 2 + (-arr_end[1]) / bev_res)
    start_bevy = int(bev_img.shape[0] - centroid[0] / bev_res)
    end_bevy = int(bev_img.shape[0] - arr_end[0] / bev_res)

    arrowedLine(bev_img, (start_bevx, start_bevy), (end_bevx, end_bevy), color, 1)

# ==============================================================================
# DRAWS A KEYPOINT IN BEV-BOX CENTER
# ==============================================================================
def draw_bbox_keypoint(bev_img: NDArray[uint8], centroid: float, cls: int, bev_res: float, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    # Convert the centroid position from world coordinates to BEV image coordinates
    ctr_img_x = int(bev_img.shape[1] / 2 + (-centroid[1]) / bev_res) 
    ctr_img_y = int(bev_img.shape[0] - centroid[0] / bev_res)
    # Set the circle radius based on the class
    if cls == 1: # Car
        radius = 5
        circle(bev_img, (ctr_img_x, ctr_img_y), radius, color, -1)
    elif cls == 2 or cls == 3: # Pedestrian, Cyclist
        radius = 3
        circle(bev_img, (ctr_img_x, ctr_img_y), radius, color, -1)

# ==============================================================================
# DRAWS A LABEL WITH FOLLOWING CONF. SCORE
# ==============================================================================
def draw_lbl_and_score(bev_img: NDArray[uint8], class_name: str, conf_score: float = None, centroid: Tuple[float,float] = (0.0, 0.0), bev_res: float = 0.1, 
                       color: Tuple[int, int, int] = (255, 255, 255), font_scale: float = 0.5, thickness: int = 1) -> None:
    score = f'{conf_score:.2f}' if conf_score is not None else '-.--'
    # Calculate image coordinates based on centroid and BEV resolution
    # x-coordinate: Centered and inverted
    ctr_img_x = int(bev_img.shape[1] / 2 + (-centroid[1]) / bev_res)
    # y-coordinate: Calculated from bottom (image height minus transformed y-position)  
    ctr_img_y = int(bev_img.shape[0] - centroid[0] / bev_res) 

    # Determine text sizes for label and score
    (lbl_width, lbl_height), _ = getTextSize(class_name, FONT_HERSHEY_DUPLEX, font_scale, thickness)
    (scr_width, scr_height), _ = getTextSize(score, FONT_HERSHEY_DUPLEX, font_scale, thickness)

    # Calculate label position with slight offset
    lbl_x = int(ctr_img_x - lbl_width / 2) + 22  
    lbl_y = int(ctr_img_y + lbl_height / 2) + 15 
    # Limit label coordinates within image boundaries
    lbl_x = max(0, min(lbl_x, bev_img.shape[1] - lbl_width))
    lbl_y = max(0, min(lbl_y, bev_img.shape[0] - lbl_height))

    # Calculate score position with offset 
    scr_x = int(ctr_img_x + 38) 
    scr_y = int(ctr_img_y + 22)
     # Limit score coordinates within image boundaries
    scr_x = max(0, min(scr_x, bev_img.shape[1] - scr_width))
    scr_y = max(0, min(scr_y, bev_img.shape[0] - scr_height))
  
    # Label
    putText(bev_img, class_name, (lbl_x, lbl_y), FONT_HERSHEY_DUPLEX, font_scale, color, thickness, LINE_AA)
    # Score
    putText(bev_img, str(score), (scr_x, scr_y), FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), thickness, LINE_AA)

# ==============================================================================
# DRAWS A LABEL WITH NEW LINE CONF. SCORE
# ==============================================================================
def draw_lbl_nl_score(bev_img: NDArray[uint8], class_name: str, conf_score: float = None, centroid: Tuple[float, float] = (0.0, 0.0), yaw: float = 0, 
                      bev_res: float = 0.1, color: Tuple[int, int, int] = (255, 255, 255), font_scale: float = 0.5, thickness: int = 1) -> None:
    score = f'{conf_score:.2f}' if conf_score is not None else '-.--'
    
    # Calculate image coordinates based on centroid and BEV resolution
    ctr_img_x = int(bev_img.shape[1] / 2 + (-centroid[1]) / bev_res)
    ctr_img_y = int(bev_img.shape[0] - centroid[0] / bev_res)
    
    # Determine text sizes for label and score
    (lbl_width, lbl_height), _ = getTextSize(class_name, FONT_HERSHEY_DUPLEX, font_scale, thickness)
    (scr_width, scr_height), _ = getTextSize(score, FONT_HERSHEY_DUPLEX, font_scale, thickness)
    
    # Calculate label position with slight offset based on yaw
    offset_x = int(np.cos(yaw) * 30)  # From the centroid
    offset_y = int(np.sin(yaw) * 30)  # Same for y-direction

    lbl_x = int(ctr_img_x - lbl_width / 2 + offset_x)  
    lbl_y = int(ctr_img_y + lbl_height / 2 + offset_y)
    
    # Limit label coordinates within image boundaries
    lbl_x = max(0, min(lbl_x, bev_img.shape[1] - lbl_width))
    lbl_y = max(0, min(lbl_y, bev_img.shape[0] - lbl_height))
    
    # Calculate score position with offset (you could adjust this based on yaw too if needed)
    scr_x = int(ctr_img_x - scr_width / 2 + offset_x) 
    scr_y = int(ctr_img_y + scr_height / 2 + 15)
    scr_x = max(0, min(scr_x, bev_img.shape[1] - scr_width))
    scr_y = max(0, min(scr_y, bev_img.shape[0] - scr_height))

    # Label
    putText(bev_img, class_name, (lbl_x, lbl_y), FONT_HERSHEY_DUPLEX, font_scale, color, thickness, LINE_AA)
    # Score
    putText(bev_img, str(score), (scr_x, scr_y), FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), thickness, LINE_AA)

# ==============================================================================
# DRAWS A ROTATED 2D-BOUNDING BOX ONTO A BEV IMAGE (BOX IN LiDAR FRAME REQUIRED)
# ==============================================================================
def get_rot_bevbox(x: float, y: float, l: float, w: float, yaw_lidar: float, cls: int, bev_img: NDArray[uint8], bev_res: float = 0.1
                   ) -> Union[Tuple[float, float, float, float, float, float, float, float, List[int], List[float]], Tuple[int, int, int, int]]:
    """
    source: https://github.com/AlejandroBarrera/birdnet2/blob/5ceed811b289796d7d7420a064ecb079c80801ab/tools/convert_kitti_to_coco_rotation.py#L73C10-L73C11
    other worthwhile sources: https://github.com/maudzung/Complex-YOLOv4-Pytorch/blob/master/src/data_process/kitti_bev_utils.py, 
    https://github.com/spmallick/learnopencv/blob/master/3D-LiDAR-Object-Detection/sfa/data_process/kitti_bev_utils.py#L87
    """
    # Convert BEV image to array and extract dimensions
    bev_img = array(bev_img)
    bvrows, bvcols, _ = bev_img.shape
    # Define object's centroid, adjust rotation angle
    centroid = [x, y]

    if cls == 1: # Car
        # CAR
        l = l + 0.4
        w = w + 0.4
    elif cls == 2: # Pedestrian
        # https://github.com/maudzung/Complex-YOLOv4-Pytorch/blob/master/src/data_process/kitti_bev_utils.py
        # PEDESTRIAN; Ped. and Cyc. labels are very small, so lets add some factor to height/width
        l = l + 0.3
        w = w + 0.3
    elif cls == 3: # Cyclist
        # CYCLIST
        l = l + 0.3
        w = w + 0.3

    yaw_bev = -yaw_lidar # Invert yaw from LiDAR frame (CW) to Image frame (CCW), bc zero angle (0°) aligned differently in both systems
    
    # Calculate the initial coordinates of the object's four corners (relative to the centroid)
    corners = array([[centroid[0] - l/2., centroid[1] + w/2.], # Top-left
                    [centroid[0] + l/2., centroid[1] + w/2.], # Top-right
                    [centroid[0] + l/2., centroid[1] - w/2.], # Bottom-right
                    [centroid[0] - l/2., centroid[1] - w/2.]]) # Bottom-left

    # Compute rotation matrix for yaw angle
    cos, sin = np.cos(yaw_bev), np.sin(yaw_bev)
    """
    2D-Rotation matrix
    [x'] = [cos(yaw) -sin(yaw)] * [x]
    [y']   [sin(yaw)  cos(yaw)]   [y]
    """
    R = array([[cos, -sin], [sin, cos]])

    # Rotate all corners around the centroid
    rotated_corners = dot(corners - centroid, R) + centroid

    # Convert the world coordinates to BEV image coordinates
    x1, x2, x3, x4 = bvcols / 2 + (-rotated_corners[:, 1]) / bev_res  # lidar world y -> image x (u)
    y1, y2, y3, y4 = bvrows - rotated_corners[:, 0] / bev_res         # lidar world x -> image y (v)

    # Now swap the corners to match YOLOv8 OBB format:
    """     Λ Default                Λ YOLOv8 OBB CW
    (x2,y2)---(x3,y3)        (x4,y4)---(x1,y1)
       |    |    |              |    |    |
       |    x    |      -->     |    x    |
       |         |              |         |
    (x1,y1)---(x4,y4)        (x3,y3)---(x2,y2)
    """
    x1, x2, x3, x4 = x3, x4, x1, x2
    y1, y2, y3, y4 = y3, y4, y1, y2

    ############
    # Option 1 #
    ############
    # The check whether at least one corner point is outside is removed.
    # NOTE: KITTI readme: "...to avoid false positives - detections not visible on the image plane should be filtered."
    # is_fully_visible = not (
    #     (x1 < 0 or x1 >= bvcols) or
    #     (x2 < 0 or x2 >= bvcols) or
    #     (x3 < 0 or x3 >= bvcols) or
    #     (x4 < 0 or x4 >= bvcols) or
    #     (y1 < 0 or y1 >= bvrows) or
    #     (y2 < 0 or y2 >= bvrows) or
    #     (y3 < 0 or y3 >= bvrows) or
    #     (y4 < 0 or y4 >= bvrows)
    # )

    ############
    # Option 2 #
    ############
    # The check whether all corner points are outside is removed.
    is_fully_outside = (
        (x1 < 0 and x2 < 0 and x3 < 0 and x4 < 0) or  
        (x1 >= bvcols and x2 >= bvcols and x3 >= bvcols and x4 >= bvcols) or 
        (y1 < 0 and y2 < 0 and y3 < 0 and y4 < 0) or 
        (y1 >= bvrows and y2 >= bvrows and y3 >= bvrows and y4 >= bvrows)  
    )

    # Clipping of Bounding-Box Coordinates
    x1 = max(0, min(x1, bvcols - 1))
    y1 = max(0, min(y1, bvrows - 1))
    x2 = max(0, min(x2, bvcols - 1))
    y2 = max(0, min(y2, bvrows - 1))
    x3 = max(0, min(x3, bvcols - 1))
    y3 = max(0, min(y3, bvrows - 1))
    x4 = max(0, min(x4, bvcols - 1))
    y4 = max(0, min(y4, bvrows - 1))

    # NOTE: Future implementation could be with clipping. When all corners of the box are outside the BEV image clip them.
    # Remove objects outside the BEV image
    # if x1 <= 0 and x2 <= 0 or \
    #    x1 >= bvcols-1 and x2 >= bvcols-1 or \
    #    y1 <= 0 and y2 <= 0 or \
    #    y1 >= bvrows-1 and y2 >= bvrows-1:
    #     return -1, -1, -1, -1  # Out of bounds
    # # Clip boxes to the BEV image
    # x1 = max(0, x1)
    # y1 = max(0, y1)
    # x2 = min(bvcols-1, x2)
    # y2 = min(bvrows-1, y2)
    
    # Calculate ROI of box
    x_min = max(0, int(min(x1, x2, x3, x4)))
    x_max = min(bvcols - 1, int(max(x1, x2, x3, x4)))
    y_min = max(0, int(min(y1, y2, y3, y4)))
    y_max = min(bvrows - 1, int(max(y1, y2, y3, y4)))

    # Remove objects with fewer than 3 points in the box
    roi = bev_img[y_min:y_max, x_min:x_max]
    """
    (x1,y1)------(x2,y1)
    |              |
    |      ROI     |
    |              |
    (x1,y2)------(x2,y2)
    """
    nonzero = count_nonzero(sum(roi, axis=2))

    if nonzero < 3:  # Detection is considered unreliable with fewer than 3 points
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, box_colormap[int(0)], [0, 0]

    # Option 1
    #if is_fully_visible:
        #return cls, x1, y1, x2, y2, x3, y3, x4, y4, box_colormap[int(cls)], centroid # Return the coordinates of the four corners of the rotated bounding box in BEV image space
    #else:
        #return array([-1, -1, -1, -1]) # Indicates the box should not be drawn (out of bounds)
        #return -1, -1, -1, -1, -1, -1, -1, -1, -1, box_colormap[int(0)], [0, 0]
    
    # Option 2
    if is_fully_outside:
        #return array([-1, -1, -1, -1]) # Indicates the box should not be drawn (out of bounds)
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, box_colormap[int(0)], [0, 0] 
    else:
        return cls, x1, y1, x2, y2, x3, y3, x4, y4, box_colormap[int(cls)], centroid



