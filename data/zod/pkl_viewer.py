import pickle

#path = '/home/rlab10/OpenPCDet/data/zod/zod_infos_train_full.pkl'
path = '/home/rlab10/OpenPCDet/data/zod/zod_val_dataset.pkl'
#path = '/home/rlab10/OpenPCDet/data/zod/zod_infos_trainval_full.pkl'
#path = '/home/rlab10/OpenPCDet/data/zod/zod_train_dataset.pkl'
#path = '/media/rlab10/Dataset/zod/zod_train_test_names.pkl'

with open(path, 'rb') as f:
    zod_infos = pickle.load(f)

list_zod = list(zod_infos)
first_entry  = zod_infos[1]
print(first_entry)

# for idx, info in enumerate(list_zod):
#     print(f"\n{'='*40} Sample Index: {idx} {'='*40}\n")
#     for key, value in info.items():
#          print(f" '{key}': {value},")
#     print(info[0]['frame_id'])
#     print(info)
#     # used for kitti_infos_val_proc.pkl or kitti_infos_train.pkl
#     #print(info['annos']['name'])
#     #print(info['annos']['gt_boxes_lidar'])
#     print("\n")


