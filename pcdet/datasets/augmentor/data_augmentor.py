from functools import partial

import numpy as np
from PIL import Image
import copy
import torch

from ...utils import common_utils
from . import augmentor_utils, database_sampler
from ...ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from ..augmentor.augmentor_utils import get_points_in_box
from ...utils.box_utils import remove_points_in_boxes3d


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        #print('DataAugmentor initialized successfully') # DEBUG

        self.data_augmentor_queue = []
        self.applied_augmentors = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST # read the augmentation config list from yaml

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST: # read the disabling augmentation list from yaml
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg) # read the names of the disabled augmentation from yaml
            self.data_augmentor_queue.append(cur_augmentor)
            self.applied_augmentors.append(cur_cfg.NAME)

    def disable_augmentation(self, augmentor_configs):
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
    
    # https://github.com/traveller59/second.pytorch/blob/1b2b58bec1c535a06d7785043664c0fc2ee375f9/second/core/sample_ops.py#L14
    # https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/augmentor/database_sampler.py     
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        print('DataAugmentor: gt_sampling() called for initializing DataBaseSampler')
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            print('DataAugmentor: random_world_flip() called with no data_dict')
            return partial(self.random_world_flip, config=config)
        
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            # overwrite random choice (after BEVDetNet paper) in augmentor_utils.py
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, return_flip=True
            )
            data_dict['flip_%s'%cur_axis] = enable
            
            print(f"DataAugmentor: flip along {cur_axis}-axis, always enabled: {enable}")
            
            if 'roi_boxes' in data_dict.keys():
                num_frame, num_rois,dim = data_dict['roi_boxes'].shape
                roi_boxes, _, _ = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                data_dict['roi_boxes'].reshape(-1,dim), np.zeros([1,3]), return_flip=True, enable=enable
                )
                data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        print('DataAugmentor: random world flip completed')
        return data_dict
    
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            print('DataAugmentor: random_world_rotation() called with no data_dict')
            return partial(self.random_world_rotation, config=config)
        
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot, enable = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )
        print(f'DataAugmentor: rotate with range: {rot_range}, enabled: {enable}')
        print(f'DataAugmentor: applied world noise rotation: {noise_rot}')
        if 'roi_boxes' in data_dict.keys():
            num_frame, num_rois,dim = data_dict['roi_boxes'].shape
            roi_boxes, _, _ = augmentor_utils.global_rotation(
            data_dict['roi_boxes'].reshape(-1, dim), np.zeros([1, 3]), rot_range=rot_range, return_rot=True, noise_rotation=noise_rot)
            data_dict['roi_boxes'] = roi_boxes.reshape(num_frame, num_rois,dim)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_glob_rot'] = noise_rot
        print('DataAugmentor: random world rotation completed')
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        
        if 'roi_boxes' in data_dict.keys():
            gt_boxes, roi_boxes, points, noise_scale = augmentor_utils.global_scaling_with_roi_boxes(
                data_dict['gt_boxes'], data_dict['roi_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )
            data_dict['roi_boxes'] = roi_boxes
        else:
            gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
                data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], return_scale=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_scale'] = noise_scale
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        assert len(noise_translate_std) == 3
        noise_translate = np.array([
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[2], 1),
        ], dtype=np.float32).T

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        points[:, :3] += noise_translate
        gt_boxes[:, :3] += noise_translate
                
        if 'roi_boxes' in data_dict.keys():
            data_dict['roi_boxes'][:, :3] += noise_translate
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_translate'] = noise_translate
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            print('DataAugmentor: random_local_rotation() called with no data_dict')
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        print(f'DataAugmentor: random local rotation with range: {rot_range}, enabled: {enable}')
        print(f'DataAugmentor: applied local noise rotation: {noise_rot}')

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['noise_loc_rot'] = noise_rot
        print('DataAugmentor: random local rotation completed')

        return data_dict

    def random_local_rotation_v2(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using. Modified version of random_local_rotation.
        """
        
        if data_dict is None:
            print('DataAugmentor: random_local_rotation_v2() called with no data_dict')
            return partial(self.random_local_rotation_v2, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        num_gt_boxes = len(data_dict['gt_boxes'])
        print('DataAugmentor: no. gt_boxes in sample: ', num_gt_boxes)
        collision_count = 0
       
        if num_gt_boxes < 2: # only one bbox in gt_boxes
            print(f'DataAugmentor: {num_gt_boxes} 3D-BB in dict entry. Skip boxes_iou3d_gpu(), applying random local rotation')
            
            gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range)
            
            print(f'DataAugmentor: random local rotation with range: {rot_range}, enabled: {enable}')
            
            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            data_dict['noise_loc_rot'] = noise_rot
            print('DataAugmentor: random local rotation completed')
            return data_dict

        else: # several bboxes in gt_boxess
            import itertools
            print(f'DataAugmentor: {num_gt_boxes} 3D-BB in dict entry. Check all IoU with boxes_iou3d_gpu()')
            
            gt_boxes = data_dict['gt_boxes']
            gt_indices = list(range(gt_boxes.shape[0]))
            iou_matrix = torch.zeros((len(gt_indices), len(gt_indices)), dtype=torch.float32).cuda()
        
            # IoU for all box pairs (excluding itself)
            for box_a_idx, box_b_idx in itertools.combinations(gt_indices, 2):
                boxes_a = torch.tensor(gt_boxes[box_a_idx:box_a_idx + 1], dtype=torch.float32).cuda()
                boxes_b = torch.tensor(gt_boxes[box_b_idx:box_b_idx + 1], dtype=torch.float32).cuda()
                
                iou3d = boxes_iou3d_gpu(boxes_a, boxes_b)
                iou_matrix[box_a_idx, box_b_idx] = iou3d # save results (idx, idx)
            
            overlap_matrix = (iou_matrix > 0).fill_diagonal_(False) # True or False in no._boxes x no._boxes matrix
            overlapping_indices = torch.nonzero(overlap_matrix) # box pairs (i, j)
           
            if overlapping_indices.numel() > 0:  # collision between 3D-BB in scene
                collision_count += overlapping_indices.size(0)
                #print(f"DataAugmentor: detected {collision_count} collision between boxes. Retrying rotation on this pair(s)...")
                print(f"DataAugmentor: detected ->{collision_count}<- collision between boxes. Continue local rotation...")
                #overlapping_boxes_set = set()

                #for idx in overlapping_indices:
                #    box_a_idx, box_b_idx = idx[0].item(), idx[1].item()
                #    overlapping_boxes_set.add(box_a_idx)
                #    overlapping_boxes_set.add(box_b_idx)
                
                #overlapping_boxes = np.array([gt_boxes[i] for i in overlapping_boxes_set], dtype=np.float32)
                #non_overlapping_indices = [i for i in gt_indices if i not in overlapping_boxes_set]
                #non_overlapping_boxes = np.array([gt_boxes[i] for i in non_overlapping_indices], dtype=np.float32)
                #points = data_dict['points']
                #points_in_overlap = np.empty((0, points.shape[1]))
                
                # extract point cloud in 3D-BB with IoU
                #for gt_box in overlapping_boxes:
                #    box_points, _ = get_points_in_box(points, gt_box)
                #    points_in_overlap = np.vstack((points_in_overlap, box_points))
                
                # try second rotation
                #gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
                #    overlapping_boxes, points_in_overlap, rot_range=rot_range)
                #print(f'DataAugmentor: local rotation applied to overlapping boxes_idx: {box_a_idx} and {box_b_idx}')
                #print(f'DataAugmentor: 2. random local rotation with range: {rot_range}, enabled: {enable}')
                
                # prepare type and check again IoU
                #gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32).cuda() # required for boxes_iou3d_gpu
                #iou_matrix_after_rotation = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[0]), dtype=torch.float32).cuda()
                
                #for box_a_idx, box_b_idx in itertools.combinations(range(gt_boxes.shape[0]), 2):
                #    boxes_a = gt_boxes[box_a_idx:box_a_idx + 1]
                #    boxes_b = gt_boxes[box_b_idx:box_b_idx + 1]
                #    iou3d_after_rotation = boxes_iou3d_gpu(boxes_a, boxes_b)
                #    iou_matrix_after_rotation[box_a_idx, box_b_idx] = iou3d_after_rotation
                #overlap_exists = (iou3d_after_rotation.item() > 0)

                #if overlap_exists:
                #    print("DataAugmentor: overlap still exists after second rotation")
                #    gt_boxes = gt_boxes[::2].cpu().numpy()
                #    print(f"DataAugmentor: removed every second bbox of each overlapping pair to resolve overlap")
                                    
                #    all_boxes = np.vstack((gt_boxes, non_overlapping_boxes)).astype(np.float32)
                #    # deactivated because of BUG
                #    data_dict['gt_boxes'] = all_boxes
                #    data_dict['points'] = points
                #    data_dict['noise_loc_rot'] = noise_rot

                #    print('DataAugmentor: local rotation on overlapping boxes completed')
                #    print(f'DataAugmentor: Total collisions detected: {collision_count}')
                #    return data_dict
                
                gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
                    data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range)
                print(f'DataAugmentor: random local rotation with range: {rot_range}, enabled: {enable}')

                # no overlap exists
                data_dict['gt_boxes'] = gt_boxes
                data_dict['points'] = points
                data_dict['noise_loc_rot'] = noise_rot

                print('DataAugmentor: local rotation on overlapping boxes completed')
                return data_dict
            
            else:
                print(f'DataAugmentor: no overlaps found between the {num_gt_boxes}-BBs. Applying random local rotation')
                gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
                    data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range)
                print(f'DataAugmentor: random local rotation with range: {rot_range}, enabled: {enable}')
                
                data_dict['gt_boxes'] = gt_boxes
                data_dict['points'] = points
                data_dict['noise_loc_rot'] = noise_rot

                print('DataAugmentor: random local rotation completed')
                return data_dict

    # dummy function
    def random_local_rotation_v3(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            print('DataAugmentor: random_local_rotation_v2() called with no data_dict')
            return partial(self.random_local_rotation_v3, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        # change here list check to different one
        if isinstance(data_dict, list): # sample with several orig. bb's
            original_gt_boxes = np.vstack([entry['gt_boxes'] for entry in data_dict if 'gt_boxes' in entry])
        else: # sample with one orig. bb
            original_gt_boxes = data_dict['gt_boxes']  
        if original_gt_boxes.shape[0] == 1:
            print('DataAugmentor: only one single 3D-Bounding Box detected. Skip box_collision_test() and apply random local rotation')
            gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
                data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
            )
            print(f'DataAugmentor: random local rotation with range: {rot_range}, enabled: {enable}')

            data_dict['gt_boxes'] = gt_boxes
            data_dict['points'] = points
            data_dict['noise_loc_rot'] = noise_rot
            print('DataAugmentor: random local rotation without box_collision_test() completed')
            return data_dict
        
        else:
            print('DataAugmentor: several 3D-Bounding Boxes detected. Run box_collision_test()')
            coll_mat_before = augmentor_utils.box_collision_test(original_gt_boxes, original_gt_boxes)
            
            # no collisions detected before rotation
            if not coll_mat_before.any():
                gt_boxes, points, noise_rot, enable = augmentor_utils.local_rotation(
                    original_gt_boxes, data_dict['points'], rot_range=rot_range
                )
                print(f'DataAugmentor: random local rotation v2 with range: {rot_range}, enabled: {enable}')

                # perform new test on augmented boxes
                coll_mat_after = augmentor_utils.box_collision_test(gt_boxes, gt_boxes)
                # Mark the diagonal of the collision matrix (where each box is compared to itself) as False to avoid false positives.
                diag_rotated = np.arange(gt_boxes.shape[0])
                coll_mat_after[diag_rotated, diag_rotated] = False
            
                valid_boxes = []
                for i in range(gt_boxes.shape[0]):
                    # if the current bounding box does not collide with any other box.
                    if not coll_mat_after[i].any():
                        valid_boxes.append(gt_boxes[i])

                data_dict['gt_boxes'] = np.array(valid_boxes)
                data_dict['points'] = points
                data_dict['noise_loc_rot'] = noise_rot
                
            else:
                print('Warning: collisions detected after rotation! Reduce angle to a proper value e.g. 9° or 10°')
        
            print('DataAugmentor: random local rotation v2 with box_collision_test() completed')
            return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def imgaug(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.imgaug, config=config)
        imgs = data_dict["camera_imgs"]
        img_process_infos = data_dict['img_process_infos']
        new_imgs = []
        for img, img_process_info in zip(imgs, img_process_infos):
            flip = False
            if config.RAND_FLIP and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*config.ROT_LIM)
            # aug images
            if flip:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            img = img.rotate(rotate)
            img_process_info[2] = flip
            img_process_info[3] = rotate
            new_imgs.append(img)

        data_dict["camera_imgs"] = new_imgs
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # changes to original_dict will have no effect on data_dict and vice versa.
        original_dict = copy.deepcopy(data_dict)

        # pre-processing for original dict
        original_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            original_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in original_dict:
            original_dict.pop('calib')
        if 'road_plane' in original_dict:
            original_dict.pop('road_plane')
        if 'gt_boxes_mask' in original_dict:
            gt_boxes_mask = original_dict['gt_boxes_mask']
            original_dict['gt_boxes'] = original_dict['gt_boxes'][gt_boxes_mask]
            original_dict['gt_names'] = original_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in original_dict:
                original_dict['gt_boxes2d'] = original_dict['gt_boxes2d'][gt_boxes_mask]

            original_dict.pop('gt_boxes_mask')

        # list to collate the original dict and the N augmented dicts
        dict_list = [original_dict]

        # augmentation loop, mentioned techniques in .yaml applied
        # Note: gt_sampling for all via if else statement
        for cur_augmentor in self.data_augmentor_queue: 
            
            temp_dict = cur_augmentor(data_dict=copy.deepcopy(data_dict))
            
            temp_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                temp_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
            if 'calib' in temp_dict:
                temp_dict.pop('calib')
            if 'road_plane' in temp_dict:
                temp_dict.pop('road_plane')
            if 'gt_boxes_mask' in temp_dict:
                gt_boxes_mask = temp_dict['gt_boxes_mask']
                temp_dict['gt_boxes'] = temp_dict['gt_boxes'][gt_boxes_mask]
                temp_dict['gt_names'] = temp_dict['gt_names'][gt_boxes_mask]
                
                # BUG: when deleting gt_boxes, mismatch gt_boxes <-> gt_boxes_mask 
                # if len(gt_boxes_mask) > temp_dict['gt_boxes'].shape[0]:
                #     gt_boxes_mask = gt_boxes_mask[:temp_dict['gt_boxes'].shape[0]]
                #     temp_dict['gt_boxes'] = temp_dict['gt_boxes'][gt_boxes_mask]
                # if len(gt_boxes_mask) > temp_dict['gt_names'].shape[0]:
                #     gt_boxes_mask = gt_boxes_mask[:temp_dict['gt_names'].shape[0]]
                #     temp_dict['gt_names'] = temp_dict['gt_names'][gt_boxes_mask]
                if 'gt_boxes2d' in temp_dict:
                    temp_dict['gt_boxes2d'] = temp_dict['gt_boxes2d'][gt_boxes_mask]

                temp_dict.pop('gt_boxes_mask')

            # add the augmented dicts to the original list
            dict_list.append(temp_dict) 

        # return the expanded list
        return dict_list, self.applied_augmentors
