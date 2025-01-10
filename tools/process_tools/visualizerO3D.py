from pcdet.datasets.kitti.kitti_dataset import *
from pcdet.datasets.dataset import *
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils, calibration_kitti, box_utils

import yaml
from easydict import EasyDict
from pathlib import Path
import pickle

import open3d 
from tools.visual_utils import open3d_vis_utils_O3D as O3DVisualizer


dataset_cfg=EasyDict(yaml.safe_load(open('/home/rlab10/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml')))
class_names=['Car', 'Pedestrian', 'Cyclist']
file_path = '/home/rlab10/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py' 
# /home/rlab10/OpenPCDet
ROOT_DIR = (Path(file_path).resolve().parent / '../../../').resolve()
data_path = ROOT_DIR / 'data' / 'kitti'
ext = '.bin'
data = []
num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, data_processor_flag=False, fov_mode=False):
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
        self.split = self.dataset_cfg.DATA_SPLIT['train']
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.demo_infos = []
        self.include_kitti_data(mode='train')
        self.data_processor_flag = data_processor_flag
        self.fov_mode = fov_mode

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('DemoDataset: Loading raw KITTI dataset')
        demo_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]: # 'train', bc training=True
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            # Read the data infos from kitti_infos_train.pkl
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                demo_infos.extend(infos)
        # Add the newly loaded KITTI dataset information to kitti_infos list.
        self.demo_infos.extend(demo_infos)

    def __len__(self):
        return len(self.sample_id_list)
    
    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

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

    def prepare_demo_data(self, data_dict):
        print('DatasetTemplate: prepare_demo_data() called')
        if 'gt_boxes' not in data_dict:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for demo visualization'
        else:         
            if data_dict.get('gt_boxes', None) is not None:
                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
                data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
                data_dict['gt_names'] = data_dict['gt_names'][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                # already transformed 3D boxes LiDAR coord. + add number for the class
                gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes'] = gt_boxes


            if data_dict.get('points', None) is not None:
                data_dict = self.point_feature_encoder.forward(data_dict)
            
            if self.data_processor_flag:
                print('DatasetTemplate: data processing activated')
                data_dict = self.data_processor.forward(data_dict=data_dict)
    
        return data_dict

    def __getitem__(self, index):
        print('DemoDataset: __getitem__ called')
        if self.merge_all_iters_to_one_epoch:
            index = index % len(self.demo_infos)
        # works
        info = copy.deepcopy(self.demo_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name = 'DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[...,np.newaxis]], axis = 1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

        input_dict.update({
            'gt_names': gt_names,
            'gt_boxes': gt_boxes_lidar
        })

        if 'points' in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.fov_mode:
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    print('DemoDataset: fov mode activated')
                    # only the points in the FOV of the camera are important for me
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]
            input_dict['points'] = points

        data_dict = self.prepare_demo_data(data_dict=input_dict)

        return data_dict

def main(mode=str):
    logger = common_utils.create_logger()
    if mode == 'raw':
        logger.info('-----------------Quick Visualizer Demo of raw data-------------------------')
        logger.info(f'Mode for visualization: {mode}')
        demo_dataset = DemoDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=data_path, logger=logger, data_processor_flag=False, fov_mode=False)
        logger.info(f'Total number of samples: \t{len(demo_dataset)}')
        # raw data not  0-4
        #demo_dataset.demo_infos[4]
        #demo_dataset[0]
        # get_infos oder prepare_data hier implementieren(flow: include_kitti_data->__getitem__->prepare_data)
        # or using KittiDataset for accessing the
        data_dict = demo_dataset[0]
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']

        O3DVisualizer.draw_demo_scenesO3D(points=points, gt_boxes=gt_boxes, point_colors=None, draw_origin=True)
        logger.info('Demo visualization of raw data done.')
        
    elif mode == 'processed':
        logger.info('-----------------Quick Visualizer Demo of pre-processed data-------------------------')
        logger.info(f'Mode for visualization: {mode}')
        demo_dataset = DemoDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=data_path, logger=logger, data_processor_flag=True, fov_mode=True, vc=None)
        logger.info(f'Total number of samples: \t{len(demo_dataset)}')
       
        data_dict = demo_dataset[2]
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes']

        O3DVisualizer.draw_demo_scenesO3D(points=points, gt_boxes=gt_boxes, point_colors=None, draw_origin=True)
        logger.info('Demo visualization of pre-processed data done.')

if __name__ == '__main__':
    main(mode='raw')
    #main(mode='processed')