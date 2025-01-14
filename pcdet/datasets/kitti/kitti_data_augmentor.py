from pcdet.datasets.kitti.kitti_dataset import *
from pcdet.datasets.dataset import *
import yaml
from easydict import EasyDict
from pathlib import Path
import os
from pcdet.utils import common_utils
from time import sleep

# absolute paths, unflexible
#dataset_cfg=EasyDict(yaml.safe_load(open('/home/rlab10/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml')))
#file_path = '/home/rlab10/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py'

ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
file_path = ROOT_DIR / 'pcdet' / 'datasets' / 'kitti' / 'kitti_dataset.py'
data_path = ROOT_DIR / 'data' / 'kitti'
save_path = ROOT_DIR / 'data' / 'kitti'

dataset_cfg_path = ROOT_DIR / 'tools' / 'cfgs' / 'dataset_configs' / 'kitti_dataset.yaml'
class_names=['Car', 'Pedestrian', 'Cyclist']
dataset_cfg = EasyDict(yaml.safe_load(open(dataset_cfg_path)))

kitti_infos = []
num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    from time import sleep
    
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, logger=common_utils.create_logger())
    
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_%s_dataset.pkl' % val_split)
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('\n' + '-' * 36 + 'Start to generate data infos' + '-' * 37)
    print('---------------CAUTION: Source code is configured to serve as Augmentor NOT training-----------------')

    dataset.set_split(train_split)
    # ensure that get_infos() processes the single scene
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, num_features=num_features)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s\n' % train_filename)
    sleep(3)

    dataset.set_split(val_split)
    # ensure that mode 'test' will process the single scene with additional postprocessing
    dataset.training = False
    kitti_val_dataset = dataset.get_infos_val(num_workers=workers, has_label=True, count_inside_pts=True, num_features=num_features)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_val_dataset, f)
    print('Kitti info val file is saved to %s\n' % val_filename)
    sleep(3)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
       pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)
    sleep(3)

    print('\n---------------Start creating groundtruth database for later data augmentation-------------------------')
    print('---------------CAUTION: Source code is configured to serve as Augmentor NOT training-------------------')
    print('---------------No DataProcessor and PointFeatureEncoder required, handled by training data creation----')
    
    # Input the 'kitti_infos_train.pkl' to generate gt_database (cutted objects of samples)
    user_input = input("\nDo you want to continue with the groundtruth database creation? (y/n): ").strip().lower()

    if user_input == 'y':
        print('Continuing the script for the creation of the database with gt_samples.\n')
        sleep(1)
        dataset.set_split(train_split)
        dataset.create_groundtruth_database(train_filename, split=train_split)
    elif user_input == 'n':
        print("WARNING: The groundtruth database is necessary for gt_sampling augmentation technique. Don't skip this or you won't be able to apply gt_sampling on your dataset.")
        exit()
    else:
        print("Invalid input. Please enter 'y' for Yes or 'n' for No.")

    print(f'---------------These groundtruth {train_split} objects are randomly inserted into samples (augmentation)-------')
    print('-' * 41 + 'Data preparation Done' + '-' * 41)

def save_data_list(data_list=None, save_path=save_path, root_path=None, sample_id_list=None, workers=4):
    root_path = Path(os.getenv('HOME')) / 'OpenPCDet' / 'data' / 'kitti'
    split = dataset_cfg.DATA_SPLIT['train']
    split_dir = root_path / 'ImageSets' / (split + '.txt')
    sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
    

    train_split = 'train'
    train_filename = save_path / ('kitti_infos_%s_augmented.pkl' % train_split)

    aug_config_list = dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    print('\n' + '-' * 29 + 'Start to save data infos(original+augmented)' + '-' * 30)

    with open(train_filename, 'wb') as f:
        pickle.dump(data_list, f)
        
    for sample_idx in sample_id_list:
        applied_augmentations = [aug_cfg['NAME'] for aug_cfg in aug_config_list]
        aug_str = ', '.join(applied_augmentations)
        print(f"{split} sample_idx: {sample_idx} (original, {aug_str})")
        print('%s sample_idx: %s' % (split, sample_id_list))
 
    print('Kitti info train/aug file is saved to %s' % train_filename)
    print('-' * 43 + 'Data saving Done' + '-' * 44 + '\n') 


if __name__ == '__main__':

    create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4)
    sleep(3)

    # not necessary
    #dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=True)
    user_input = input("\nWould you like to continue with the database creation? (y/n): ").strip().lower()
    
    if user_input == 'y':
        print('Continuing the script for the creation of the database with data augmentation.\n')
        sleep(1)
        dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=True)
        dataset.dataset_w_all_infos = dataset.get_infos(num_workers=4, has_label=True, count_inside_pts=True, num_features=num_features)

        dataset_as_list = []

        for idx in range(len(dataset)):
            dataset_as_list.append(dataset[idx])

        save_data_list(data_list=dataset_as_list, save_path=save_path, root_path=None, sample_id_list=None, workers=4)
    elif user_input == 'n':
        print("WARNING: The groundtruth database is necessary for gt_sampling augmentation. Don't skip this or you won't be able to create the dataset.")
        exit()
    else:
        print("Invalid input. Please enter 'y' for Yes or 'n' for No.")
    

