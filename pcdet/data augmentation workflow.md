# KITTI Data Augmentation

1. Run the modified `create_kitti_infos()` function.
    * this will create kitti_infos_train.pkl, kitti_val_dataset.pkl and kitti_infos_test.pkl.
    * kitti_infos_train.pkl will be used for creating the groundtruth database. 
    * kitti_val_dataset.pkl is the validation dataset, which will be used later in BEV image creation.
    * kitti_infos_test.pkl is not used in my case, retained due to the standard workflow of OpenPCDet.

    * Internally the function `create_groundtruth_database()` will create the folder `gt_database` with extracted samples from kitti_infos_train.pkl. The extracted samples will be used for later data augmentation when technique `gt_sampling` is applied.