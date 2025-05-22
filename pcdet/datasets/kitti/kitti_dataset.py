import copy
import pickle

import numpy as np
from skimage import io
from pathlib import Path

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        #print('Class KittiDataset: initializing.') # DEBUG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # Reading training sample names.
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.dataset_w_all_infos = [] # new list for custom training dataset
        self.kitti_infos = []
        self.include_kitti_data(self.mode)
       

    def include_kitti_data(self, mode):
        print('KittiDataset: loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]: # 'train', bc training=True
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            # Read the data infos from kitti_infos_train.pkl
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        # Add the newly loaded KITTI dataset information to kitti_infos list.
        self.kitti_infos.extend(kitti_infos)
        print('KittiDatset: Total training samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        """
            Loads point cloud for a sample
                Args: 
                    index (int): Index of the point cloud file to get.
                Returns:
                    np.array(N, 4): point cloud
        """
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate.
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag
    
    # Modified version of get_infos() for pre-processing validation data
    def get_infos_val(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None, num_features=4, class_names=None):
        import concurrent.futures as futures

        if self.mode == 'test': # val mode
            from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
            #from pcdet.datasets.processor.data_processor import DataProcessor 

            point_feature_encoder = PointFeatureEncoder(self.dataset_cfg.POINT_FEATURE_ENCODING, point_cloud_range=self.point_cloud_range)

            # Note: DONT use data_processor, it filters out necessary object details
            #data_processor = DataProcessor(self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, 
            #                               training=self.training, num_point_features=self.point_feature_encoder.num_point_features)
            
            def process_single_scene_val(sample_idx):
                print('%s sample_idx: %s' % (self.split, sample_idx))
                print('Mode: %s, for pre-procoessing (DataProcessor & PointFeatureEncoder) validation data' % self.mode)
                info = {}
                pc_info = {'num_features': num_features, 'lidar_idx': sample_idx} # num_features: x,y,z,intensity
                info['point_cloud'] = pc_info

                image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
                info['image'] = image_info
                calib = self.get_calib(sample_idx)

                P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)  
                R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
                R0_4x4[3, 3] = 1.
                R0_4x4[:3, :3] = calib.R0
                V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
                calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
                
                info['calib'] = calib_info

                if has_label:
                    obj_list = self.get_label(sample_idx) # Object3d kitti
                    
                    # 13.11.2024 custom: filter the obj_list related to class_names
                    filtered_obj_list = [obj for obj in obj_list if obj.cls_type in class_names]
                    annotations = {}

                    # 13.11.2024 default: containing all object classes from KITTI
                    #annotations['name'] = np.array([obj.cls_type for obj in obj_list]) # label names
                    
                    # 13.11.2024 custom: containing only specified classes of filtered_obj_list
                    annotations['name'] = np.array([obj.cls_type for obj in filtered_obj_list]) # label names
                    annotations['truncated'] = np.array([obj.truncation for obj in filtered_obj_list]) # truncation of object
                    annotations['occluded'] = np.array([obj.occlusion for obj in filtered_obj_list]) # occulsion of object
                    annotations['alpha'] = np.array([obj.alpha for obj in filtered_obj_list]) # object angle in image
                    annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in filtered_obj_list], axis=0) # xmin, ymin, xmax, ymax
                    annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in filtered_obj_list])  # l,h,w (camera) format
                    annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in filtered_obj_list], axis=0) # x,y,z (camera) format
                    annotations['rotation_y'] = np.array([obj.ry for obj in filtered_obj_list]) # heading y-axis (camera) format
                    annotations['score'] = np.array([obj.score for obj in filtered_obj_list]) # confidence in detection
                    annotations['difficulty'] = np.array([obj.level for obj in filtered_obj_list], np.int32)
                    # 13.11.2024 default: all object classes (even Truck, Van etc.) in the label are counted, except 'DontCare'
                    #num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                    # 13.11.2024 custom: only allowed_classes object classes in the label are counted
                    num_objects = len(filtered_obj_list)
                    num_gt = len(annotations['name'])
                    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                    annotations['index'] = np.array(index, dtype=np.int32)
                    # only the gt_boxes are inserted, which base on num_objects
                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                    loc_lidar = calib.rect_to_lidar(loc)
                    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    loc_lidar[:, 2] += h[:, 0] / 2
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                    # 13.11.2024 custom: index added to gt_boxes (Car: 1, Pedstrian: 2, Cyclist: 3)
                    gt_classes = np.array([class_names.index(n) + 1 for n in annotations['name']], dtype=np.int32)
                    gt_boxes_lidar = np.concatenate((gt_boxes_lidar, gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                    
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar

                    info['annos'] = annotations

                    if count_inside_pts:
                        points = self.get_lidar(sample_idx)
                        calib = self.get_calib(sample_idx)
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                        pts_fov = points[fov_flag]

                        # this entry is needed for point feature encoding.forward()
                        info['points'] = pts_fov # default only FoV points
                        #info['points'] = points # all points in scene

                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                        for k in range(num_objects):
                            flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                            num_points_in_gt[k] = flag.sum()
                        annotations['num_points_in_gt'] = num_points_in_gt

                        #temp_info = point_feature_encoder.forward(data_dict=info)
                        info = point_feature_encoder.forward(data_dict=info)    
                        
                        # Note: DONT use data_processor.forward(data_dict=temp_info), it filters out necessary object details
                        #info = data_processor.forward(data_dict=temp_info)

                return info 
            
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        if self.mode == 'test': # val mode
            with futures.ThreadPoolExecutor(num_workers) as executor:
                infos = executor.map(process_single_scene_val, sample_id_list)
            return list(infos)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures
            
        def process_single_scene(sample_idx):
            print('KittiDataset: %s sample_idx: %s' % (self.split, sample_idx))
            print('KittiDataset: pre-processing during data augmentation via __getitem__')

            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx} # num_features: x,y,z,intensity
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)  
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            
            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}

                # TODO: Ultralytics YOLOv8 OBB
                # Der Objektdetektor kann ja nicht anhand occluded, truncated, difficulty unterscheiden?
                # Lsg: manuel splitting of categories, then performing mAP.

                # Add filter to eval.py in ultralytics repo for
                # Easy: Min. bounding box height: 40 Px = <40 Px rauswerfen (Länge object in BEV rausfiltern) 
                # Moderate: Min. bounding box height: 25 Px = <25 Px rauswerfen
                # Hard: Min. bounding box height: 25 Px = < 25 Px rauswerfen
                
                # Brauche ich das Label in Kamera Koordinaten oder LiDAR für den Objektdetektor aus der BEV?
                # Lsg: gt_boxes sind LiDAR Koordinaten projiziere in BEV frame; Lsg: Kamera koords. vom kamera frame sind ausreichend.
                # Reminder: label_cam: [x, y, z, h, w, l, r]      label_lidar: [x, y, z, dx, dy, dz, heading]
                # Approach: label_lidar: [x, y, z, dx, dy, dz, heading]  label_bev: [bev_x, bev_y, l, w]???
                # https://github.com/TRAILab/CaDDN/blob/5a96b37f16b3c29dd2509507b1cdfdff5d53c558/pcdet/utils/box_utils.py#L261
                
                # Ultralytics YOLOv8 OBB calculates internal rotation_y, compare later if kitti label (heading)
                # is more precise or calculation algorithm.

                # Note: here it would be possible to only keep classes that are interesting for training
                annotations['name'] = np.array([obj.cls_type for obj in obj_list]) # label name
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list]) # truncation of object
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list]) # occulsion of object
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list]) # object angle in image
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0) # xmin, ymin, xmax, ymax
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # l,h,w (camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0) # x,y,z (camera) format
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list]) # heading y-axis (camera) format
                annotations['score'] = np.array([obj.score for obj in obj_list]) # confidence in detection
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                # Note: adjust this here too, to keep desired classes
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
        
            return info
       
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
    
    # Create gt database for data augmentation
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)
        readme_file = database_save_path / 'README.md'
        readme_content="""
        Information about gt_database and gt_sampling:

        This directory stores cutted out ground truth objects from samples to insert them randomly in the current sample during training.

        From the scientific paper "SECOND: Sparsely Embedded Convolutional Detection":
        "Sample Ground Truths from the Database
        The major problem we encountered during training was the existence of too few ground truths,
        which significantly limited the convergence speed and final performance of the network. To solve this
        problem, we introduced a data augmentation approach. First, we generated a database containing
        the labels of all ground truths and their associated point cloud data (points inside the 3D bounding
        boxes of the ground truths) from the training dataset. Then, during training, we randomly selected
        several ground truths from this database and introduced them into the current training point cloud
        via concatenation. Using this approach, we could greatly increase the number of ground truths per
        point cloud and simulate objects existing in different environments. To avoid physically impossible
        outcomes, we performed a collision test after sampling the ground truths and removed any sampled
        objects that collided with other objects."
        """

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        # Open 'kitti_infos_train.pkl'
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # Write readme file
        with open(readme_file, 'w') as f:
            f.write(readme_content)

        # For each .bin file
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k+1, len(infos)))
            
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            print('  => contains cut-outs sample from %s.bin' % sample_idx)
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        # Output the num of all classes in database                
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v))) # frequency of a class in all samples

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
            print('KittiDataset: kitti db info file is saved to %s' % db_info_save_path)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl (lidar -> camera)

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs and self.logger.info('kitti infos: %s', self.kitti_infos)

        return len(self.kitti_infos)

    def __getitem__(self, index):
        print('KittiDataset:  __getitem__ called')
        
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        # Replace the old list against a list with all wanted infos. This leaves you free to use entries as you wish 
        # without having a preconfigured training (missing some entries).
        #info = copy.deepcopy(self.kitti_infos[index])
        info = copy.deepcopy(self.dataset_w_all_infos[index])
    
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape'] 
        calib = self.get_calib(sample_idx) 
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points']) 
        
        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            # loc=x, y, z, dims=h, w, l, rots=ry
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # x, y, z, h, w, l, ry
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # transforms 3D Bounding Box from Camera coordinates to 3D Bounding Box in LiDAR coordinates
            # [x, y, z, h, w, l, ry] -> [x, y, z, dx(l), dy(w), dz(h), heading]
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list: # only for the model CaDDN relevant
                input_dict['gt_boxes2d'] = annos["bbox"]

            # Road planes are used for gt_sampling augmentation, where we could 'paste' the additional GTs to the road plane of current scene.
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                # only the points in the FOV of the camera are important for me
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "images" in get_item_list: # only for the model CaDDN relevant
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list: # only for the model CaDDN relevant
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list: # only for the model CaDDN relevant
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        
        # change the parameter data_dict to data for working with several dicts instead of one
        # here the data_infos are changed
        data_list, applied_augmentors = self.prepare_data(data=input_dict) # jump to dataset.py
        
        if 'annos' in info:
            # loop over data_dicts in data_list (camera frame information)
            for data_dict in data_list: 
                data_dict['truncated'] = annos['truncated']
                data_dict['occluded'] = annos['occluded']
                data_dict['alpha'] = annos['alpha']
                data_dict['dimensions'] = annos['dimensions']
                data_dict['location'] = annos['location']
                data_dict['rotation_y'] = annos['rotation_y']
                data_dict['difficulty'] = annos['difficulty']
                data_dict['image_shape'] = img_shape

        return data_list, applied_augmentors

    # template from custom_dataset.py, maybe useful
    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{name} {x} {y} {z} {dx} {dy} {dz} {angle}\n".format(
                    name=name, x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6]
                )
                f.write(line) 

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    from time import sleep
    # dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False) # Default
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, logger=common_utils.create_logger())
     
    train_split, val_split = 'train', 'val'
    # Replacement for the fix value of 4 in get_infos() method
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    
    # Not needed, because training on evaluation needs grount truth labels.
    # Test would be just for inference, no metrics can be evaluated.
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('\n---------------Start to generate data infos--------------------------------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, num_features=num_features)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s\n' % train_filename)
    sleep(3)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, num_features=num_features)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)
    sleep(3)
    
    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s\n' % trainval_filename)
    sleep(3)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
       pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s\n' % test_filename)
    sleep(3)

    print('\n---------------Start create groundtruth database for later data augmentation---------------')
    print('------These groundtruth objects are randomly inserted into scenes during training----------')
    
    # Input the 'kitti_infos_train.pkl' to generate gt_database (cutted objects of samples)
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------\n')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':

        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
