import pickle

#path = '/home/rlab10/OpenPCDet/data/kitti/kitti_train_dataset.pkl'
#path = '/home/rlab10/OpenPCDet/data/kitti/kitti_infos_train.pkl'
#path = '/home/rlab10/OpenPCDet/data/kitti/kitti_val_dataset.pkl'
#path = '/home/rlab10/OpenPCDet/data/kitti/unchanged pkl files after gt_database/kitti_infos_val.pkl'
#path = '/home/rlab10/OpenPCDet/data/kitti/kitti_dbinfos_train.pkl'

# Domain Generalization
#path = '/home/rlab10/OpenPCDet/data/kitti/Domain Generalization/densification/kitti_train_dataset_3x_densified.pkl'
path = '/home/rlab10/OpenPCDet/data/kitti/Domain Generalization/random beam re-sampling/kitti_train_dataset_rbrs.pkl'
#path = '/home/rlab10/OpenPCDet/data/kitti/kitti_train_dataset_beamlabels.pkl'
#path = '/home/rlab10/OpenPCDet/data/kitti/Domain Generalization/random beam re-sampling/kitti_val_dataset_rbrs.pkl'

#path = "/home/rlab10/OpenPCDet/data/kitti/Domain Generalization/pdrw interpolation/kitti_val_dataset_pdrw.pkl"


with open(path, 'rb') as f:
    kitti_infos = pickle.load(f)

list_kitti = list(kitti_infos)
first_entry  = kitti_infos[0]

print(first_entry)

#print(kitti_infos)


# for idx, info in enumerate(list_kitti):
#     #print(f"\n{'='*40} Sample Index: {idx} {'='*40}\n")
#     #for key, value in info.items():
#     #     print(f" '{key}': {value},")
#     #print(info[0]['frame_id'])
#     print(info)
#     # used for kitti_infos_val_proc.pkl or kitti_infos_train.pkl
#     #print(info['annos']['name'])
#     #print(info['annos']['gt_boxes_lidar'])
#     print("\n")


