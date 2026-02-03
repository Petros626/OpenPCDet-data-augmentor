import copy
import pickle
import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_zod
from ..dataset import DatasetTemplate

from zod import ZodFrames
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject
from zod.utils.geometry import get_points_in_camera_fov, transform_points
from zod.data_classes.geometry import Pose
import zod.constants as constants


class ZODDatasetCustom(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, creating_pkl_infos=False):
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

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.version = self.dataset_cfg.DATASET_VERSION 
        self.countries =  self.dataset_cfg.DATASET_COUNTRIES
        self.kitti_calib_path = self.root_path / self.dataset_cfg.KITTI_CALIB_PATH
        self.creating_pkl_infos = creating_pkl_infos
        self.zod_frames = ZodFrames(dataset_root=self.root_path, version=self.version)

        # Create ImageSets from trainval-frames-full.json
        imagesets_dir = self.root_path / 'ImageSets'
        if not (imagesets_dir / 'train_full.txt').exists():
            self.create_ImageSets_from_zodtrainval(zod_frames=self.zod_frames, imagesets_dir=imagesets_dir, version='full', only_split='train')
        if not (imagesets_dir / 'val_full.txt').exists():
            self.create_ImageSets_from_zodtrainval(zod_frames=self.zod_frames, imagesets_dir=imagesets_dir, version='full', only_split='val')
        else:
            if self.logger is not None:
                self.logger.info(f'ImageSets already exist in {imagesets_dir.resolve()}')

        split_dir = self.root_path / 'ImageSets' / (self.split + '_' + self.version + '.txt') # ZOD
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
    
        self.dataset_w_all_infos = [] # new list for custom augmented training dataset
        self.zod_infos = []
        self.Tr_Zod_Lidar_to_Kitti_Lidar = np.array([[0, -1, 0],
                                                  [1,  0, 0], 
                                                  [0,  0, 1]]) 
     
        self.include_zod_data(self.mode)
        self.kitti_calib = self.load_kitti_calib(self.kitti_calib_path)

    def create_ImageSets_from_zodtrainval(self, zod_frames, imagesets_dir, version='full', only_split=None):
        from pathlib import Path

        # keep it updated, when you encounter:
        # FileNotFoundError: [Errno 2] No such file or directory: '/home/rlab10/OpenPCDet/data/zod/single_frames/024391/lidar_velodyne/024391_golf_2021-10-11T11:25:02.972379Z.npy'
        faulty_train_frames = ['069293', '058043', '014942', '057435', '062628', '097451', '027256', '001554', '024391',
                               '046291', '053347', '028927','056545', '009608', '061077', '054192', '056158', '008896',
                               '020452', '009277', '057300', '090283', '012439', '006494', '082269', '024304', '044369',
                               '070476', '026378', '004782', '028087', '063518', '057144', '030924', '002639', '062073',
                               '056269', '005912', '052151', '049713', '052528', '016020', '003027', '059396', '052749',
                               '000410']

        train_id_list = list(zod_frames.get_split(constants.TRAIN))
        train_id_list = [s for s in train_id_list if not any(exclude_frames in s for exclude_frames in faulty_train_frames)]
        val_id_list = list(zod_frames.get_split(constants.VAL))

        imagesets_dir = Path(imagesets_dir)
        imagesets_dir.mkdir(exist_ok=True)

        if only_split is None or only_split == 'train':
            with open(imagesets_dir / f'train_{version}.txt', 'w') as f:
                for item in train_id_list:
                    f.write(item + '\n')
            print(f'Saved train split to {imagesets_dir / f"train_{version}.txt"}')

        if only_split is None or only_split == 'val':
            with open(imagesets_dir / f'val_{version}.txt', 'w') as f:
                for item in val_id_list:
                    f.write(item + '\n')
            print(f'Saved val split to {imagesets_dir / f"val_{version}.txt"}')

    def include_zod_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading ZOD dataset')
        zod_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]: # 'train', bc training=True
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            # Read the data infos from zod_infos_train_full.pkl
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                zod_infos.extend(infos)

        self.zod_infos.extend(zod_infos)
        if self.logger is not None:
            self.logger.info('Total samples for ZOD dataset: %d' % (len(zod_infos)))
        
        if not self.creating_pkl_infos:
            self. map_merged_classes()
        
    def map_merged_classes(self):
        if self.dataset_cfg.get('MAP_MERGED_CLASSES', None) is None:
            return
        
        # update class names in zod_infos
        map_merge_class = self.dataset_cfg.MAP_MERGED_CLASSES
        for info in self.zod_infos:
            assert 'annos' in info
            info['annos']['name'] = np.vectorize(lambda name: map_merge_class[name], otypes=[str])(info['annos']['name'])

    def set_split(self, split, version):
        import random

        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.version = version

        split_dir = self.root_path / 'ImageSets' / (self.split + '_' + self.version + '.txt')
        all_ids = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []

        if self.countries and len(self.countries) > 0:
           candidate_ids = [x for x in all_ids if self.zod_frames[x].metadata.country_code in self.countries]

        else:
           candidate_ids = all_ids

        random_seed = self.dataset_cfg.SPLIT_CONFIG.RANDOM_SEED
        random.seed(random_seed)

        if split == 'train' and not self.dataset_cfg.SPLIT_CONFIG.USE_ORIGINAL_SPLIT:
            train_size = self.dataset_cfg.SPLIT_CONFIG.TRAIN_SIZE
            valid_ids = []
            checked_ids = set()
            while len(valid_ids) < train_size and len(checked_ids) < len(candidate_ids):
                candidate = random.choice(candidate_ids)
                if candidate in checked_ids:
                    continue
                checked_ids.add(candidate)
                if self.get_label(candidate) is not None: # only valid train samples from train_full.txt
                    valid_ids.append(candidate)
            self.sample_id_list = valid_ids
            if self.logger is not None:
                self.logger.info(f"{len(self.sample_id_list)} valid samples for {self.countries} in split {self.split}")

        elif split == 'val' and not self.dataset_cfg.SPLIT_CONFIG.USE_ORIGINAL_SPLIT:
            val_size = self.dataset_cfg.SPLIT_CONFIG.VAL_SIZE
            random_seed = self.dataset_cfg.SPLIT_CONFIG.RANDOM_SEED
            random.seed(random_seed)

            valid_ids = []
            checked_ids = set()
            for candidate in candidate_ids:
                checked_ids.add(candidate)
                if self.get_label(candidate) is not None: # only valid labels from val_full.txt
                    valid_ids.append(candidate)

            if len(valid_ids) < val_size:
                train_split_dir = self.root_path / 'ImageSets' / ('train_' + self.version + '.txt')
                train_ids = [x.strip() for x in open(train_split_dir).readlines()] if train_split_dir.exists() else []
                train_ids_unique = [x for x in train_ids if x not in checked_ids]
                random.shuffle(train_ids_unique)
                for candidate in train_ids_unique:
                    if len(valid_ids) >= val_size:
                        break
                    if self.get_label(candidate) is not None: # only valid labels from train_full.txt
                        valid_ids.append(candidate)
            self.sample_id_list = valid_ids
            if self.logger is not None:
                self.logger.info(f"{len(self.sample_id_list)} valid samples for {self.countries} in split {self.split}")

        else:
            self.sample_id_list = candidate_ids
            if self.logger is not None:
                self.logger.info(f"{len(self.sample_id_list)} samples for (all countries) in split {self.split}")

    def get_lidar(self, idx, num_features=4):
        """
            Loads point cloud for a sample
                Args: 
                    index (int): Index of the point cloud file to get.
                Returns:
                    np.array(N, 4): point cloud
        """
        try:
            zod_frames_files = self.zod_frames[idx]
            lidar_core_frame = zod_frames_files.info.get_key_lidar_frame()
            pc = lidar_core_frame.read()
            # filter LiDAR source to contain only VLS128 [0, 128), not LiDAR beams fron the 2x VLP-16 [128, 144), [144, 160)
            if self.dataset_cfg.get('USE_VLS128_ONLY', False): 
                vls128_mask = pc.diode_idx < 128
                pc.points = pc.points[vls128_mask]
                pc.intensity = pc.intensity[vls128_mask]
                pc.diode_idx = pc.diode_idx[vls128_mask]
                pc.timestamps = pc.timestamps[vls128_mask]
        except Exception as e:
            print(f"Error loading Lidar for {idx}: {e}")

        if num_features == 4:
            # scale intensity to [0,1] from [0,255], bc at ZOD it isn't default
            pc.intensity = pc.intensity / 255
            # (x, y, z, intensity)
            points = np.concatenate((pc.points, pc.intensity.reshape(-1,1)), dtype=np.float32, axis=1)
        elif num_features == 5:
            pc.intensity = pc.intensity / 255
            points = np.concatenate((pc.points, pc.intensity.reshape(-1, 1), pc.diode_idx.reshape(-1, 1)), dtype=np.float32, axis=1)
        elif num_features == 3:
            points = pc.points
        else:
            raise NotImplementedError

        return points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        zod_frames_files = self.zod_frames[idx]
        camera_core_frame = zod_frames_files.info.get_key_camera_frame(Anonymization.BLUR)
        image = camera_core_frame.read()
        image = image.astype(np.float32)
        image /= 255.0

        return image
    
    def get_image_shape(self, idx):
        """
        Returns image shape (height, width) for a sample using ZOD API metadata.
        Args:
            idx: int, Sample index
            anonymization: str, 'BLUR' or 'DNAT'
        Returns:
            np.array([height, width], dtype=np.int32)
        """
        zod_frames_files = self.zod_frames[idx]
        camera_core_frame = zod_frames_files.info.get_key_camera_frame(Anonymization.BLUR)
        image = camera_core_frame.read()
        
        return np.array(image.shape[:2], dtype=np.int32)
    
    def get_fov_flag(self, pts_lidar, calib, camera=Camera.FRONT, lidar=Lidar.VELODYNE, use_kitti_fov=False):
        """
        Args:
            points (np.ndarray): LiDAR points in ZOD LiDAR coordinate system, shape (N, 3)
            calib: ZOD calibration object with camera and LiDAR extrinsics
            camera: ZOD Camera Enum (default: FRONT)
            lidar: ZOD Lidar Enum (default: VELODYNE)

        Returns:
            np.ndarray: Boolean mask (N,), True for point in FoV of camera
        """
        # Transformation LiDAR -> World
        t_lidar_to_world = calib.lidars[lidar].extrinsics
        # Transformation Camera -> World
        t_camera_to_world = calib.cameras[camera].extrinsics

        # Transformation World -> Camera
        t_world_to_camera= t_camera_to_world.inverse
        # Combine transformations LiDAR -> World -> Camera
        t_lidar_to_camera = Pose(t_world_to_camera.transform @ t_lidar_to_world.transform)

        points_img = transform_points(pts_lidar, t_lidar_to_camera.transform)

        # Only points with positive
        positive_depth = points_img[:, 2] > 0 # z>0
        mask = np.zeros(pts_lidar.shape[0], dtype=bool)
        if not np.any(positive_depth):
            return mask

        points_img_valid = points_img[positive_depth]

        if use_kitti_fov:
            kitti_fov = self.dataset_cfg.KITTI_FOV # (90°, 35°)
            _, fov_mask = get_points_in_camera_fov(fov=kitti_fov, camera_data=points_img_valid) # KITTI
        else:
            zod_fov = calib.cameras[camera].field_of_view # (120°, 67°)
            _, fov_mask = get_points_in_camera_fov(fov=zod_fov, camera_data=points_img_valid) # ZOD

        mask[positive_depth] = fov_mask

        return mask

    # source: https://github.com/griesbchr/3DTrans/blob/3174699105aefb3ed11e524606f707fd91239850/pcdet/datasets/zod/zod_dataset.py#L224
    def get_object_truncation_binary(self, corners, calib):
        corners_flat = corners.reshape(-1,3)
        
        # check if all corners are in Camera FoV
        # points in camera mask
        mask = self.get_fov_flag(corners_flat, calib)

        # reshape mask to per box shape
        mask = mask.reshape(-1,8) # 8x 3D cornes

        # if all corners are in FoV, object is not truncated
        truncated = np.zeros(mask.shape[0], dtype=float)
        truncated[mask.sum(axis=1) < 8] = 1.0

        return truncated # 0: non-truncated, 1: truncated
    
    def get_object_truncation_wrong(self, box3d, calib, camera=Camera.FRONT, fov_range=245.0):
        """
            4 -------- 5            0 -------- 1         \_            _/
           /|         /|            |          |   obj 3 |\|  Camera  |/| obj 2
          7 .------- 6 .            |          |           \  FoV    /
          | |   3D   | |    -->     |    BEV   |    -->     \ 120° _/
          . 0 -------. 1            | (Camera) |             \    |/| obj 1
          |/ (LiDAR) |/             |          |              \   /
          3 -------- 2              3 -------- 2               \ /
        """
        from shapely.geometry import Polygon
      
        # Create a copy and transform to camera frame using ZOD API
        box3d_camera = box3d.copy() # BOX3D (Camera) <- BOX3D (LiDAR)
        box3d_camera.convert_to(camera, calib)

        # Extract BEV corners in camera frame
        bev_corners_camera = box3d_camera.corners_bev
        
        box_polygon = Polygon(bev_corners_camera)

        fov = calib.cameras[camera].field_of_view
        horizontal_fov, _ = fov # (~120°, ~67°)

        num_points = 200
        angles = np.linspace(-horizontal_fov/2, horizontal_fov/2, num_points) * np.pi / 180 # [-60°...60°]
        fov_points = np.stack([fov_range * np.sin(angles), fov_range * np.cos(angles)], axis=1)
        fov_polygon = Polygon(np.vstack([[0, 0], fov_points])) # sector in BEV

        intersection = box_polygon.intersection(fov_polygon)
        visible_area = intersection.area # A in m²
        total_area = box_polygon.area

        # KITTI like truncation: Proportion outside = 1 - proportion visible
        truncation = 1.0 - (visible_area / total_area) if total_area > 0 else 1.0

        return truncation # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries

    # source: https://github.com/zenseact/EdgeAnnotationZChallenge/blob/35afb0dcffd6b7ca3982a9a3ffbe50e9c92875f0/eval/convert_annotations_to_kitti.py#L102
    def get_object_truncation(self, box2d, image_shape):
        """
        Calculates the KITTI-style truncation value for a 2D bounding box.

        KITTI definition:
            "Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries."

        Args:
            box2d: 2D bounding box object with .xmin, .ymin, .xmax, .ymax, and .area attributes
            image_shape: (height, width) of the image as a numpy array

        Returns:
            truncation: Float in [0, 1], where 0 means fully inside the image, 1 means fully outside.
        """
        img_w, img_h = image_shape[1], image_shape[0]
        box_area = box2d.area
       
        x_min_clipped = np.clip(box2d.xmin, 0, img_w)
        x_max_clipped = np.clip(box2d.xmax, 0, img_w)
        y_min_clipped = np.clip(box2d.ymin, 0, img_h)
        y_max_clipped = np.clip(box2d.ymax, 0, img_h)

        visible_area = max(0, x_max_clipped - x_min_clipped) * max(0, y_max_clipped - y_min_clipped)

        if box_area == 0:
            return 1.0  # truncated
        
        # Truncation is the proportion of the box area outside the imagee
        truncation = 1.0 - (visible_area / box_area)
        # Ensure truncation is within [0, 1]
        return np.clip(truncation, 0.0, 1.0)

    def get_label(self, idx): 
        zod_frame = self.zod_frames[idx]
        obj_list = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION) # contains all necessary information
        image_shape = self.get_image_shape(idx)

        # NOTE: Objects visible both in the camera image and the LiDAR
        # point cloud are also labeled with a 9-DOF 3D bounding box,
        # described by the coordinate of the center of the box, length,
        # width, height size, and the four quaternion rotation parameters
        # of the cuboid (i.e., qw, qx, qy, and qz).
        # -> filtering after box2d and box3d may result in some annotations outside the camera FoV, but visible in LiDAR (less annos)
        # -> filtering after box3d only may result in more annotations, bc visible in LiDAR.

        # filter out objects only without 3d anno
        #obj_list = [obj for obj in obj_list if obj.box3d is not None]

        # filter out objects without 2d&3d anno
        obj_list = [obj for obj in obj_list if obj.box2d is not None and obj.box3d is not None]

        # DEBUG print
        #if self.logger is not None:
        #    self.logger.info("filtered out %d objects without 3d anno" % (len(zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)) - len(obj_list)))
        #if len(obj_list) == 0:
        #    return None
        
        # filter out objects that are not in class_names
        # only Bicycles with_rider = True flag
        if self.class_names is not None:
            obj_list = [obj for obj in obj_list if obj.subclass in self.class_names and
                        (obj.subclass != "VulnerableVehicle_Bicycle" or obj.with_rider == True)] # Vehicle_Car, Pedestrian, VulnerableVehicle_Bicycle (with_rider:True)
        if len(obj_list) == 0:
            return None # skip empty samples
    
        # calculate truncation and insert occlusion from ZOD
        for obj in obj_list:
            # old wrong
            #obj.truncation = self.get_object_truncation_wrong(box3d=obj.box3d, calib=zod_frame.calibration, fov_range=self.dataset_cfg.POINT_CLOUD_RANGE[3])
            
            obj.truncation = self.get_object_truncation(box2d=obj.box2d, image_shape=image_shape)
            obj.occlusion = object3d_zod.zod_occlusion_to_kitti(obj.occlusion_level)

            # NOTE: If an object is fully seen through another vehicle’s windows it should have occlusion level None.
                # ZOD: None: 0%, Light: 1% - 20%, Medium: 21% - 50%, Heavy: 51% - 80%, VeryHeavy: 81% - 100%
                # KITTI: 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
                # Mapping:
                    # ZOD 'None' → KITTI 0 (fully visible)
                    # ZOD 'Light', 'Medium' → KITTI 1 (partly occluded)
                    # ZOD 'Heavy', 'VeryHeavy' → KITTI 2 (largely occluded)
                    # other (e.g. unknown) → KITTI 3 (unknown)

            obj.level = object3d_zod.get_zod_obj_level(obj)

        # alternative future implementation for alpha, source: https://github.com/AlejandroBarrera/birdnet2/blob/5ceed811b289796d7d7420a064ecb079c80801ab/tools/val_net_BirdNetPlus.py 

        # Project points to camera frame coordinates
        # calib = Calibration(calib_file)
        # p = calib.project_velo_to_rect(np.array([[obj3d.location.x,obj3d.location.y,obj3d.location.z]]))
        
        # Obtain alpha from yaw
        #obj3d.alpha = obj3d.yaw - (-math.atan2(p[0][2], p[0][0]) - 1.5*math.pi)
        #obj3d.alpha = obj3d.alpha % (2*math.pi)
        #if obj3d.alpha > math.pi:
        #    obj3d.alpha -= 2*math.pi
        #elif obj3d.alpha < -math.pi:
        #    obj3d.alpha += 2*math.pi

        return obj_list

    def get_calib(self, idx):
        zod_frame = self.zod_frames[idx]
        
        return zod_frame.calibration
    
    def load_kitti_calib(self, calib_path):
        calib = {}

        with open(calib_path, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                values = [float(x) for x in value.strip().split()]
                calib[key.strip()] = values

        return {'P2': calib['P2'], 'R0_rect': calib['R0_rect'], 'Tr_velo_to_cam': calib['Tr_velo_to_cam']}

    # Modified version of original get_infos() for pre-processing validation data
    def get_infos_val(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures
        import time

        #if self.mode == 'test': # validation mode
        from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
        
        point_feature_encoder = PointFeatureEncoder(self.dataset_cfg.POINT_FEATURE_ENCODING, point_cloud_range=self.point_cloud_range)

        from pcdet.datasets.processor.data_processor import DataProcessor 
        data_processor = DataProcessor(self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, 
                                        training=self.training, num_point_features=self.point_feature_encoder.num_point_features)

        def process_single_scene_val(sample_idx):
            print('ZODDatasetCustom: %s sample_idx: %s' % (self.split, sample_idx))

            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx} # num_features: x,y,z,intensity
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info

            calib = self.get_calib(sample_idx)
            camera = Camera.FRONT
            lidar = Lidar.VELODYNE

            # ZOD
            calib_info = {
                'cam_intrinsics': calib.cameras[camera].intrinsics.tolist(), # 3x3
                'cam_extrinsics': calib.cameras[camera].extrinsics.transform.tolist(), # 4x4 
                'cam_distortion': calib.cameras[camera].distortion.tolist(), # 4 vector
                'cam_field_of_view': calib.cameras[camera].field_of_view.tolist(), # horizontal, vertical (degress)
                'cam_undistortion': calib.cameras[camera].undistortion.tolist() , # 4 vector
                'lidar_extrinsics': calib.lidars[lidar].extrinsics.transform.tolist(), #   
            }
            # KITTI
            calib_info['kitti_P2'] = self.kitti_calib['P2']
            calib_info['kitti_R0_rect'] = self.kitti_calib['R0_rect']
            calib_info['kitti_Tr_velo_to_cam'] = self.kitti_calib['Tr_velo_to_cam'] 

            info['calib'] = calib_info

            if not has_label:
                return info
            
            if has_label:
                obj_list = self.get_label(sample_idx)
                if obj_list is None:
                    return None # skip empty sample
                #filtered_obj_list = [obj for obj in obj_list if obj.subclass in class_names]

                annotations = {}
                                
                annotations['name'] = np.array([obj.subclass for obj in obj_list])
                if self.dataset_cfg.get('MAP_MERGED_CLASSES', None) is not None:
                    map_merge_class = self.dataset_cfg.MAP_MERGED_CLASSES
                    annotations['name'] = np.vectorize(lambda name: map_merge_class[name], otypes=[str])(annotations['name'])
                
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['bbox'] = np.array([obj.box2d.xyxy for obj in obj_list], dtype=np.float32) # xmin, ymin, xmax, ymax
                # no filtering for 2d box (see def get_label())
                # annotations['bbox'] = np.array([[-1, -1, -1, -1] for _ in obj_list], dtype=np.float32) # dummy value, not used
                annotations['dimensions'] = np.array([obj.box3d.size for obj in obj_list]) # l, w, h (LiDAR) format
                annotations['location'] = np.array([obj.box3d.center for obj in obj_list]) # x, y, z (LiDAR) format
                annotations['yaw'] = np.array([obj.box3d.orientation.yaw_pitch_roll[0] for obj in obj_list]) # rotation_z (LiDAR) format, from pyquaternion
                annotations['box3d_corners'] = np.array([obj.box3d.corners for obj in obj_list])
                annotations['score'] = np.array([-1.0 for _ in obj_list], dtype=np.float32) # dummy value, not provided
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], dtype=np.int32) # Easy, Moderate or Hard
                
                # Index array - filter out 'unclear' objects
                num_objects = len(obj_list)
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                annotations['location'] = annotations['location'] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate
                annotations['location'][:,2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift
                annotations['yaw'] = annotations['yaw'] - np.pi/2 # not + np.pi/2
                annotations['yaw'] = common_utils.limit_period(annotations['yaw'], offset=0.5, period=2 * np.pi) # [-pi, pi]

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['yaw'][:num_objects]
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                annotations.pop('yaw')

                # calculate observation angle alpha
                # source: https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_utils.py#L5
                gt_boxes_lidar = annotations['gt_boxes_lidar'].copy()
                # only relevant for model predictions and KITTI
                #gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2 
                annotations['location'][:, 0] = -gt_boxes_lidar[:, 1]  # cam_x = -y_lidar
                annotations['location'][:, 1] = -gt_boxes_lidar[:, 2]  # cam_y = -z_lidar
                annotations['location'][:, 2] = gt_boxes_lidar[:, 0]  # cam_z = x_lidar
                rotation_y = -gt_boxes_lidar[:, 6] - np.pi / 2.0 # rotation ry around Y-axis in camera coordinates
                annotations['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + rotation_y # angle betw. cam & obj. centre
                # limit alpha, rotation_y for mathematical correctness
                annotations['alpha'] = common_utils.limit_period(annotations['alpha'], offset=0.5, period=2 * np.pi) # [-pi, pi]
                rotation_y = common_utils.limit_period(rotation_y, offset=0.5, period=2 * np.pi)
                annotations['rotation_y'] = rotation_y

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx, num_features=4) # points in ZOD coordinate system (see above)

                    calib = self.get_calib(sample_idx)
                    fov_flag = self.get_fov_flag(pts_lidar=points[:, 0:3], calib=calib) # FoV filtering in ZOD coordinate system
                    pts_fov = points[fov_flag]

                    if self.dataset_cfg.FOV_POINTS_ONLY: # (120°, 67°)
                        # coordinate system alignment to KITTI
                        pts_fov[:, :3] = pts_fov[:, :3] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate points to KITTI coordinate system
                        pts_fov[:, 2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift points to KITTI coordinate system
                        info['points'] = pts_fov
                    elif self.dataset_cfg.KITTI_FOV_POINTS_ONLY: # (90°, 35°)
                        fov_flag_k = self.get_fov_flag(pts_lidar=points[:, 0:3], calib=calib, use_kitti_fov=self.dataset_cfg.KITTI_FOV)
                        pts_fov_k = points[fov_flag_k]
                        # coordinate system alignment to KITTI
                        pts_fov_k[:, :3] = pts_fov_k[:, :3] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate points to KITTI coordinate system
                        pts_fov_k[:, 2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift points to KITTI coordinate system
                        info['points'] = pts_fov_k
                    else:
                        # coordinate system alignment to KITTI
                        points[:, :3] = points[:, :3] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate points to KITTI coordinate system
                        points[:, 2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift points to KITTI coordinate system
                        info['points'] = points

                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar) # we need exact the same corner order, ZOD defines different one
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

                    temp_info = point_feature_encoder.forward(data_dict=info)
                    # Note: Use DataProcessor to limit the cloud to pc range in config file
                    info = data_processor.forward(data_dict=temp_info)

            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        #if self.mode == 'test': # val mode
        start_time = time.time()
        # improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            #infos = executor.map(process_single_scene_val, sample_id_list)
            infos = [info for info in executor.map(process_single_scene_val, sample_id_list) if info is not None]
        end_time = time.time()
        print("Total time for loading infos: ", end_time - start_time, "s")
        print("Loading speed for infos: ", len(sample_id_list) / (end_time - start_time), "sample/s")

        return list(infos)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures
        import time

        def process_single_scene(sample_idx):
            print('ZODDatasetCustom: %s sample_idx: %s' % (self.split, sample_idx))

            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info

            calib = self.get_calib(sample_idx)
            camera = Camera.FRONT
            lidar = Lidar.VELODYNE

            calib_info = {
                'cam_intrinsics': calib.cameras[camera].intrinsics.tolist(), # 3x3
                'cam_extrinsics': calib.cameras[camera].extrinsics.transform.tolist(), # 4x4 
                'cam_distortion': calib.cameras[camera].distortion.tolist(), # 4 vector
                'cam_field_of_view': calib.cameras[camera].field_of_view.tolist(), # horizontal, vertical (degress)
                'cam_undistortion': calib.cameras[camera].undistortion.tolist() , # 4 vector
                'lidar_extrinsics': calib.lidars[lidar].extrinsics.transform.tolist(), # 
            }
            
            info['calib'] = calib_info

            if not has_label:
                return info
            
            if has_label:
                obj_list = self.get_label(sample_idx)

                if obj_list is None:
                    return None # skip empty sample
                annotations = {}

                annotations['name'] = np.array([obj.subclass for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['bbox'] = np.array([obj.box2d.xyxy for obj in obj_list], dtype=np.float32) # xmin, ymin, xmax, ymax
                # no filtering for 2d box (see def get_label())
                # annotations['bbox'] = np.array([[-1, -1, -1, -1] for _ in obj_list], dtype=np.float32) # dummy value, not used
                annotations['dimensions'] = np.array([obj.box3d.size for obj in obj_list]) # l, w, h (LiDAR) format
                annotations['location'] = np.array([obj.box3d.center for obj in obj_list]) # x, y, z (LiDAR) format
                annotations['yaw'] = np.array([obj.box3d.orientation.yaw_pitch_roll[0] for obj in obj_list]) # rotation_z (LiDAR) format, from pyquaternion
                annotations['box3d_corners'] = np.array([obj.box3d.corners for obj in obj_list])
                annotations['score'] = np.array([-1.0 for _ in obj_list], dtype=np.float32) # dummy value, not provided
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], dtype=np.int32) # Easy, Moderate or Hard
              
                # Index array - filter out 'unclear' objects
                num_objects = len([obj for obj in obj_list if not obj.unclear])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                
                # rotate and shift coordinate system to match KITTI (90 deg around z axis and shift to LiDAR plane)
                """
                ZOD LiDAR Coordinate System:          KITTI LiDAR Coordinate System:
                            Z (up)                                Z (up)
                            |                                     |
                            |                                     |  Y (left)
                            |                                     | /
                            |_____Y (forward, car front)          |/_____X (forward, car front)
                           /                                     
                          /                                     
                         X (right) 
                """
                # ZOD: X-right, Y-forward, Z-up (90° rotated compared to KITTI)
                # KITTI: X-forward, Y-left, Z-up
                # Rotation matrix: 90° around Z-axis (counterclockwise)
                # [x, y, z] @ R
                # [x_kitti]   [-y_zod] * [0  -1  0]     
                # [y_kitti] = [-x_zod] * [1   0  0]   
                # [z_kitti]   [z_zod]  * [0   0  1] 

                annotations['location'] = annotations['location'] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate
                annotations['location'][:,2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift
                """
                ZOD: yaw=0 -> object shows in y-direction
                KITTI: yaw=0 -> object shows in x-direction
                Why -pi/2?:
                You rotate the coordinate system by +90°, so the yaw must be adjusted by -90° to maintain the object direction
                """
                annotations['yaw'] = annotations['yaw'] - np.pi/2 # not + np.pi/2, see above
                annotations['yaw'] = common_utils.limit_period(annotations['yaw'], offset=0.5, period=2 * np.pi) # [-pi, pi]

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['yaw'][:num_objects]
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                annotations.pop('yaw')

                # calculate observation angle alpha
                # source: https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_utils.py#L5
                """
                 ^ θl    ^ θ              
                  \     /
                 __\___/_ 
                |   \ /--|--------> 
                |____\___|
                object\     
                       \
                        \
                         \
                          \
                           \
                            \  
                             \ 
                             _\___ θray
                             \ \ /-------->
                              \ /
                            Camera

                θ: global orientation of a car
                θl: local orientiation
                θray: angle between camera view and object centre
                α = θ - θray
                """
                gt_boxes_lidar = annotations['gt_boxes_lidar'].copy()
                # only relevant for model predictions and KITTI
                #gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2 
                annotations['location'][:, 0] = -gt_boxes_lidar[:, 1]  # cam_x = -y_lidar
                annotations['location'][:, 1] = -gt_boxes_lidar[:, 2]  # cam_y = -z_lidar
                annotations['location'][:, 2] = gt_boxes_lidar[:, 0]  # cam_z = x_lidar
                rotation_y = -gt_boxes_lidar[:, 6] - np.pi / 2.0 # rotation ry around Y-axis in camera coordinates
                annotations['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + rotation_y # angle betw. cam & obj. centre
                # limit alpha, rotation_y for mathematical correctness
                annotations['alpha'] = common_utils.limit_period(annotations['alpha'], offset=0.5, period=2 * np.pi) # [-pi, pi]
                rotation_y = common_utils.limit_period(rotation_y, offset=0.5, period=2 * np.pi)
                annotations['rotation_y'] = rotation_y

                info['annos'] = annotations

                if count_inside_pts:
                    calib = self.get_calib(sample_idx)
                    points = self.get_lidar(sample_idx, num_features=4) # points in ZOD coordinate system (see above)
                    fov_flag = self.get_fov_flag(pts_lidar=points[:, 0:3], calib=calib) # FoV filtering in ZOD coordinate system
                    pts_fov_aligned_kitti = points[fov_flag]

                    # coordinate system alignment to KITTI
                    pts_fov_aligned_kitti[:, :3] = pts_fov_aligned_kitti[:, :3] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate points to KITTI coordinate system
                    pts_fov_aligned_kitti[:, 2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift points to KITTI coordinate system

                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar) # we need exact the same corner order, ZOD defines different one
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov_aligned_kitti[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        start_time = time.time()
        # improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            #infos = executor.map(process_single_scene, sample_id_list)
            infos = [info for info in executor.map(process_single_scene, sample_id_list) if info is not None]

        end_time = time.time()
        print("Total time for loading infos: ", end_time - start_time, "s")
        print("Loading speed for infos: ", len(sample_id_list) / (end_time - start_time), "sample/s")
        
        return list(infos)

    def create_groundtruth_database(self, info_path=None, version="full", used_classes=None, split='train'):
        import torch
        from pathlib import Path
        import time

        database_save_path = Path(self.root_path) / ('gt_database_%s_%s' % (split, version) if split == 'train' else ('gt_database_%s_%s' % (split, version)))
        db_info_save_path = Path(self.root_path) / ('zod_dbinfos_%s_%s.pkl' % (split, version))

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

        # Open 'zod_infos_train_full.pkl'
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # Write readme file
        with open(readme_file, 'w') as f:
            f.write(readme_content)

        start_time = time.time()

        # For each .bin file
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k+1, len(infos)))
            
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            print('  => contains cut-outs sample from %s.npy' % sample_idx)
            points = self.get_lidar(sample_idx, num_features=4) # points in ZOD coordinate system
            points[:, :3] = points[:, :3] @ self.Tr_Zod_Lidar_to_Kitti_Lidar # rotate points to KITTI coordinate system
            points[:, 2] -= self.dataset_cfg.LIDAR_Z_SHIFT # shift points to KITTI coordinate system
                        
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
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i) # leave it as .bin for OpenPCDet intern
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                if gt_points.shape[0] == 0: # Skip empty GT objects
                    continue

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

        end_time = time.time()
        print("Total time for creating gt_database: ", end_time - start_time, "s")
        print("Loading speed for gt_database: ", len(infos) / (end_time - start_time), "sample/s")

        # Output the num of all classes in database                
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v))) # frequency of a class in all samples

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
            print('ZODDatasetCustom: zod db info file is saved to %s' % db_info_save_path)

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
            # check before use!
            camera=Camera.FRONT
            pred_boxes_camera = pred_boxes.copy()
            pred_boxes_camera.convert_to(camera, calib) # box3d LiDAR --> box3d camera
            pred_boxes_img = box_utils.boxes3d_zod_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape

            )
            # only for KITTI!
            #pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            #pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #    pred_boxes_camera, calib, image_shape=image_shape
            #)

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
                    dims = single_pred_dict['dimensions'] # lhw (LiDAR) -> hwl (Camera)

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            if self.logger is not None:
                self.logger.info('zod_infos: %s', self.zod_infos)
            return len(self.zod_infos) * self.total_epochs 

        return len(self.zod_infos)
    
    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.zod_infos)
        
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
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        if self.dataset_cfg.FOV_POINTS_ONLY and self.dataset_cfg.VERTICAL_FOV_ONLY:
                raise ValueError("Configuration errorr: Only one of FOV_POINTS_ONLY or VERTICAL_FOV_ONLY may be true!")
        
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx, num_features=4) # points in ZOD coordinate system
            if self.dataset_cfg.FOV_POINTS_ONLY:
                fov_flag = self.get_fov_flag(pts_lidar=points[:, 0:3], calib=calib) # hor. & ver. FoV filtering in ZOD coordinate system
                points = points[fov_flag]
            if self.dataset_cfg.KITTI_FOV_POINTS_ONLY:
                fov_flag = self.get_fov_flag(pts_lidar=points[:, 0:3], calib=calib, use_kitti_fov=self.dataset_cfg.KITTI_FOV) # only vertical FoV filtering in ZOD coordinate system
                points = points[fov_flag]

            points[:, :3] = points[:, :3] @ self.Tr_Zod_Lidar_to_Kitti_Lidar
            points[:, 2] -= self.dataset_cfg.LIDAR_Z_SHIFT

            input_dict['points'] = points

        input_dict['calib'] = calib

        input_dict['cam_info'] = {
            'truncated': annos['truncated'],
            'occluded': annos['occluded'],
            'alpha': annos['alpha'],
            'bbox': annos['bbox'],
            'dimensions': annos['dimensions'],
            'location': annos['location'],
            'rotation_y': annos['rotation_y'],
            'score': annos['score'],
            'difficulty': annos['difficulty'],
            'image_shape': img_shape
        }

        data_list, applied_augmentors = self.prepare_data_custom(data=input_dict) # jump to dataset.py

        return data_list, applied_augmentors

def create_zod_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    from time import sleep

    dataset = ZODDatasetCustom(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, logger=common_utils.create_logger(), creating_pkl_infos=True)
   
    train_split, val_split = 'train', 'val'
    version = 'full'

    # Replacement for the fix value of 4 in get_infos() method
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('zod_infos_%s_%s.pkl' % (train_split, version))
    val_filename = save_path / ('zod_%s_dataset.pkl' % val_split)
    trainval_filename = save_path / ('zod_infos_trainval_%s.pkl' % version)

    print('\n' + '-' * 36 + 'Start to generate data infos' + '-' * 37)
    print('---------------CAUTION: Custom code is configured to serve as Augmentor NOT training-----------------')

    dataset.set_split(train_split, version)
    zod_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, num_features=num_features)
    with open(train_filename, 'wb') as f:
        pickle.dump(zod_infos_train, f)
    print('Zod info train file is saved to %s\n' % train_filename)
    sleep(3)

    dataset.set_split(val_split, version)
    # ensure that mode 'test' will process the single scene with PointFeatureEncoder, DataProcessor, FOV_FLAG
    dataset.training = False
    zod_val_dataset = dataset.get_infos_val(num_workers=workers, has_label=True, count_inside_pts=True, num_features=num_features, class_names=class_names)
    with open(val_filename, 'wb') as f:
        pickle.dump(zod_val_dataset, f)
    print('Zod info val file is saved to %s' % val_filename)
    sleep(3)
    
    with open(trainval_filename, 'wb') as f:
        pickle.dump(zod_infos_train + zod_val_dataset, f)
    print('Zod info trainval file is saved to %s\n' % trainval_filename)
    sleep(3)

    print('\n---------------Start creating groundtruth database for later data augmentation-------------------------')
    print('---------------CAUTION: Custom code is configured to serve as Augmentor NOT training-------------------')
    print('---------------No DataProcessor and PointFeatureEncoder required, handled by training data creation----')

    # Input the 'zod_infos_train_full.pkl' to generate gt_database (cutted objects of samples)
    dataset.set_split(train_split, version)
    dataset.create_groundtruth_database(info_path=train_filename, version=version, used_classes=class_names, split=train_split)
    print(f'---------------These groundtruth {train_split} objects are randomly inserted into samples (augmentation)-------')
    print('-' * 41 + 'Data preparation Done' + '-' * 41)

if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_zod_infos':

        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_zod_infos(
            dataset_cfg=dataset_cfg,
            class_names = ['Vehicle_Car', 'Pedestrian', 'VulnerableVehicle_Bicycle'],
            data_path=ROOT_DIR / 'data' / 'zod',
            save_path=ROOT_DIR / 'data' / 'zod'
        )

# wenn alles passt die ZOD-Daten im KITTI-Format als .txt erstellen (Evaluierung)?

    
    

      



    