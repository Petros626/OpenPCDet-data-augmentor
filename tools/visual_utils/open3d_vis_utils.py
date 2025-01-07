"""
Open3d visualization tool box
Written by Jihan YANG
Modified by Petros626
All rights preserved from 2021 - present.
"""
import open3d
import open3d.visualization
import open3d.ml
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1], # not assigned
    [0, 1, 0], # Car Green
    [1, 0, 1], # Pedestrian Violet
    [1, 1, 0], # Cyclist Yellow
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1)) # default blue

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def draw_demo_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, vc=None, use_custom_diagonal_color=False):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window(width=640, height=480)
  
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin (x = Forward, y = Left ,z = Upward)
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3]) # x, y, z
    vis.add_geometry(pts)

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_demo_box(vis, gt_boxes, keypoint_color=(0, 0, 1)) 

    view_control = vis.get_view_control()

    if vc == 'val':
        params = open3d.io.read_pinhole_camera_parameters('ScreenCamera_val.json')
        view_control.convert_from_pinhole_camera_parameters(params,allow_arbitrary=True)
    else:
        #view_control = vis.get_view_control()
        params = open3d.io.read_pinhole_camera_parameters('ScreenCamera_all.json')
        view_control.convert_from_pinhole_camera_parameters(params,allow_arbitrary=True)
    
    vis.run()
    vis.destroy_window()
    return vis


def translate_boxes_to_open3d_instance(gt_boxes,  use_custom_diagonal_color=False):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)


    lines = np.asarray(line_set.lines)
    diagonal_lines = np.array([[1, 4], [7, 6]]) # creates diagonal cross in direction of the object
    lines = np.concatenate([lines, diagonal_lines], axis=0)
    line_set.lines = open3d.utility.Vector2iVector(lines) # 3D-BBox lines
    
    class_idx = int(gt_boxes[7])
    bbox_color = box_colormap[class_idx]
    line_colors = np.full((len(lines), 3), bbox_color)  # Red for the box
    # heading diagonal
    if use_custom_diagonal_color:
        line_colors[-2:] = [0, 1, 0]  # Green for the diagonal
    line_set.colors = open3d.utility.Vector3dVector(line_colors)

    # the middle of the diagonal showing in object direction
    box_corner_points = np.asarray(box3d.get_box_points())
    """     
             4-------- 5
           /|         /|
          6 -------- 7 .
          | |        | |
          . 0 -------- 1
          |/         |/
          2 -------- 3
    """
    # DEBUG
    # print('box_corner_points: ', box_corner_points)
    
    midpoint_1_4 = (box_corner_points[1] + box_corner_points[4]) / 2
    midpoint_7_6 = (box_corner_points[7] + box_corner_points[6]) / 2
    middle_diagonal = (midpoint_1_4 + midpoint_7_6)  / 2

    return line_set, box3d, middle_diagonal

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        #if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def create_bbox_keypoint(class_idx, center, size, color=None):
    """Create a sphere at the top of the bounding box."""
    #top_pos = center + np.array([0, 0, size[2] / 2])  # Center of the top face
    if class_idx == 1: # Car
        radius=0.5
        top_pos = center + np.array([0, 0, size[2]/2 + 0.1])
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
    elif class_idx == 2 or class_idx == 3: # Pedestrian/Cyclist
        radius=0.23
        top_pos = center + np.array([0, 0, size[2]/2 + 0.1])
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        
    sphere.translate(top_pos)
    sphere.paint_uniform_color(color)
    return sphere

def create_bbox_arrow(class_idx, position, direction, cylinder_r=None, cone_r=None, cylinder_h=None, cone_h=None, color=None):
    """Create an arrow in the middle diagonal of the object direction."""
    arrow = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius=cylinder_r, cone_radius=cone_r, cylinder_height=cylinder_h, cone_height=cone_h)

    if class_idx == 1 or class_idx == 3: # Car/Cyclist 
        arrow.rotate(open3d.geometry.get_rotation_matrix_from_xyz([0, np.pi/2, 0]), center=[0, 0, 0])
        arrow.rotate(open3d.geometry.get_rotation_matrix_from_xyz([0, 0, direction]), center=[0, 0, 0])
    elif class_idx == 2: # Pedestrian
        arrow.rotate(open3d.geometry.get_rotation_matrix_from_xyz([0, np.pi/2, 0]), center=[0, 0, 0])
        arrow.rotate(open3d.geometry.get_rotation_matrix_from_xyz([0, 0, direction]), center=[0, 0, 0])
    
    arrow.translate(position)
    arrow.paint_uniform_color(color)
    return arrow

# def create_bbox_label(scale=0.1, density=2000, offset=(0, 0, 0.1)):
#     t = np.linspace(0, 2 * np.pi, density)
#     x = 0.25 * np.cos(t) * scale 
#     y = np.sin(t) * scale  
#     z = np.zeros_like(t)

#     points_label = np.column_stack((x, y, z))
#     return points_label

def text_3d(text, pos, direction=None, degree=0.0, density=None, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=None):
    """Create a 3d text label"""
    if direction is None:
        direction = (0., 0., 1.) # default positioned to z-axis
        
    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font, font_size * density)
    left, top, right, bottom = font_obj.getbbox(text)
    width = right - left
    height = bottom - top
    font_dim = (width, height)

    img = Image.new('1', font_dim, color=1)
    draw = ImageDraw.Draw(img)
    draw.text((-left, -top), text, font=font_obj, fill=0)  # Draw black text (fill=0 for black)
    img = np.asarray(img)

    img_mask = img == 0  # Only the black pixels (text) will be considered
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T
    pcd = open3d.geometry.PointCloud()
    pcd.colors = open3d.utility.Vector3dVector(np.ones((indices.shape[0], 3)) * np.array([1, 0, 0])) 
    pcd.points = open3d.utility.Vector3dVector(indices / (1000 * density))

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    
    # Apply rotation and translation using quaternion
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)  # Translate to the specified position
    pcd.transform(trans)

    return pcd

def create_bbox_label(text, pos):
    label_pcd = text_3d(text, pos, font_size=100, density=20)
    return label_pcd

def draw_demo_box(vis, gt_boxes, ref_labels=None, keypoint_color=(0, 0, 1)):
    
    for i in range(gt_boxes.shape[0]):
        line_set, box3d, middle_diagonal = translate_boxes_to_open3d_instance(gt_boxes[i], use_custom_diagonal_color=False)

        # Draw 3D BBox lines
        vis.add_geometry(line_set)

        #if gt_labels is not None:
        #    line_set.paint_uniform_color(color)
        #else:
        #    line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        # already colored lines

        class_idx = int(gt_boxes[i, 7])
        bbox_color = box_colormap[class_idx]
        arrow_color = bbox_color
        
        # Add object keypoint
        center = gt_boxes[i, :3]
        size = gt_boxes[i, 3:6]
        keypoint = create_bbox_keypoint(class_idx, center, size, color=keypoint_color)
        vis.add_geometry(keypoint)

        # Add object arrow
        heading = gt_boxes[i, 6]
        arrow = create_bbox_arrow(class_idx, middle_diagonal, heading, cylinder_r=0.090, cone_r=0.270, cylinder_h=1.4, cone_h=0.2, color=arrow_color)
        vis.add_geometry(arrow)

        # Add object label
        #box_corner_points = box3d.get_box_points()
        #label_origin = box_corner_points[4] + 0.1
        #text = '1'
        #label = create_bbox_label(text, pos=label_origin)
        #vis.add_geometry(label)

        # label_points = create_bbox_label(scale=0.1, density=1000) + label_origin
        # label_pcd = open3d.geometry.PointCloud()
        # label_pcd.points = open3d.utility.Vector3dVector(label_points)
        # label_pcd.paint_uniform_color([0, 1, 0])
        # vis.add_geometry(label_pcd)

    return vis
