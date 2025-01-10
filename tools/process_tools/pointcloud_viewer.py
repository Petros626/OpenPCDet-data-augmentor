import numpy as np
import pptk 

# short working example
#xyz = pptk.rand(10, 3)
#v = pptk.viewer(xyz)
#v.attributes(xyz)
#v.set(point_size=0.01)

unchanged_point_cloud = '/home/rlab10/OpenPCDet/data/kitti/unchanged gt_database/000000_Pedestrian_0.bin'
changed_point_cloud = '/home/rlab10/OpenPCDet/data/kitti/custom gt_database/000000_Pedestrian_0.bin'




# works also
#point_cloud_data = np.fromfile(path_to_point_cloud, dtype=np.float32).reshape(-1, 4)
#xyz_data = point_cloud_data[:, :3]
#v = pptk.viewer(xyz_data, debug=True)
point_cloud_data = np.fromfile(changed_point_cloud, '<f4')  # little-endian float32
point_cloud_data = np.reshape(point_cloud_data, (-1, 4))    # x, y, z, r
pptk.viewer(point_cloud_data[:, :3], debug=True)
pptk.set(point_size=0.01)
pptk.clear()
pptk.close()

