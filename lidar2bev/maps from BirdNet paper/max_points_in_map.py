"""
Script for offline estimation of the ground position (BirdNet 1 only)
Detail: The script creates a 2D map (with optional 3D information through height “slices”) to estimate the maximum
point density in a space, based on the parameters of a LIDAR scan. It uses simulations of LIDAR beams to calculate 
the point density in defined cells. The map is divided into grid cells (size defined by cell_size) and stores the 
number of points in each cell.
Modified by Petros626
Source by Jorge Beltrán de la Cita (https://github.com/beltransen/lidar_bev/blob/48425014520a8bc8205f7f464df3a87b421cb233/scripts/max_points_map.py)
"""

import numpy as np
import argparse
from progress.bar import IncrementalBar
import os

map_size = 60 # Size of the BEV grid in meters (from paper BEVDetNet x = (0, 60))
cell_size = 0.1 # Size of the cells in the BEV grid (in meters) (from paper BEVDetNet δ = 0.1m)
min_height = -2.73 # Minimum height (from paper BEVDetNet z-coordinates (-2.73, 1.27))
max_height = 1.27 # Maximum height (see min_height)
num_slices = 3 # Number of height levels
num_planes = 64 # Number of LiDAR planes (see HDL64-E manual)
velo_minangle = -24.9 # Minimum horizontal angle (see HDL64-E manual)
velo_hres = 0.2 # Horizontal angle resolution (see HDL64-E manual: 0.08°(finer res.) - 0.35°(coarser res.))
velo_vres = 0.4 # Vertical angle resolution (see HDL64-E manual)
velo_height = 1.73 # LiDAR height (from paper Vision meets Robotics: The KITTI Dataset)

def get_max_points_map(bv_size, cell_size, z_min, z_max, nslices, nplanes, low_angle, h_res, v_res, lidar_height):
    grid_size = int(bv_size / cell_size)
    print('Generating map with size: {} x {} '.format(grid_size, grid_size))
    max_map = np.zeros((grid_size, grid_size, nslices), dtype=np.float64)
    hrad_res = h_res*np.pi/180
    vrad_res = v_res*np.pi/180
    lowrad_angle = low_angle*np.pi/180

    bar = IncrementalBar('Processing', max=int(grid_size/2), suffix='%(percent).1f%% - Estimated time: %(eta)ds')

    for i in range(0, int(grid_size/2)):  # i -> row (y-axis velo plane / x-axis image plane)
        for j in range(0, i+1): # j -> col (x-axis velo plane / y-axis image plane)
            # Compute plane coords
            x_min = cell_size * j
            x_max = cell_size * (j + 1)
            y_min = cell_size * i
            y_max = cell_size * (i + 1)

            max_angle = np.arctan(y_max / x_min) if x_min !=0 else np.pi/2
            change_angle = np.arctan(y_min / x_min) if x_min !=0 else np.pi/2
            min_angle = np.arctan(y_min / x_max) if x_max !=0 else 0
            num_hor_rays = (max_angle-min_angle)/hrad_res

            # if x_min == 0 and y_min == 0: # Closest cells takes all points
            #     for k in xrange(0, nslices):
            #         max_map[int(grid_size/2)-1][int(grid_size/2)-1][k]=nplanes*num_hor_rays
            #         max_map[int(grid_size/2)-1][int(grid_size/2)][k]=nplanes*num_hor_rays
            #         max_map[int(grid_size/2)][int(grid_size/2)][k]=nplanes*num_hor_rays
            #         max_map[int(grid_size/2)][int(grid_size/2)-1][k]=nplanes*num_hor_rays

            # else:
            for p in range(0, nplanes):
                current_height_angle = min_angle
                current_vngle = lowrad_angle + p * vrad_res

                # Trace ray
                # Side plane
                while current_height_angle < change_angle:
                    y = y_min

                    # fix "invalid value encountered in scalar divide"
                    epsilon = 1e-10
                    x = np.cos(current_height_angle)*y/(np.sin(current_height_angle)+epsilon)

                    distance_xy = np.sqrt(x**2+y**2)
                    distance_3d = distance_xy/np.cos(current_vngle)
                    z = lidar_height + np.sin(current_vngle)*distance_3d

                    z_slice = (z_max - z_min) / nslices
                    for k in range(0, nslices):
                        if k*z_slice < z <= (k+1)*z_slice:
                            # Increment corresponding cell
                            max_map[int(grid_size/2)-1-j][int(grid_size/2)-1-i][k]+=1
                            max_map[int(grid_size/2)-1-j][int(grid_size/2)+i][k]+=1
                            max_map[int(grid_size/2)+j][int(grid_size/2)+i][k]+=1
                            max_map[int(grid_size/2)+j][int(grid_size/2)-1-i][k]+=1

                            if i != j:
                                max_map[int(grid_size/2)-1-i][int(grid_size/2)-1-j][k]+=1
                                max_map[int(grid_size/2)-1-i][int(grid_size/2)+j][k]+=1
                                max_map[int(grid_size/2)+i][int(grid_size/2)+j][k]+=1
                                max_map[int(grid_size/2)+i][int(grid_size/2)-1-j][k]+=1

                    current_height_angle += hrad_res

                # Front plane
                while current_height_angle < max_angle:
                    x = x_min
                    y = np.sin(current_height_angle)*x/np.cos(current_height_angle)

                    distance_xy = np.sqrt(x**2+y**2)
                    distance_3d = distance_xy/np.cos(current_vngle)
                    z = lidar_height + np.sin(current_vngle)*distance_3d

                    z_slice = (z_max - z_min) / nslices
                    for k in range(0, nslices):
                        if k*z_slice < z <= (k+1)*z_slice:
                            max_map[int(grid_size/2)-1-j][int(grid_size/2)-1-i][k]+=1
                            max_map[int(grid_size/2)-1-j][int(grid_size/2)+i][k]+=1
                            max_map[int(grid_size/2)+j][int(grid_size/2)+i][k]+=1
                            max_map[int(grid_size/2)+j][int(grid_size/2)-1-i][k]+=1

                            if i != j:
                                max_map[int(grid_size/2)-1-i][int(grid_size/2)-1-j][k]+=1
                                max_map[int(grid_size/2)-1-i][int(grid_size/2)+j][k]+=1
                                max_map[int(grid_size/2)+i][int(grid_size/2)+j][k]+=1
                                max_map[int(grid_size/2)+i][int(grid_size/2)-1-j][k]+=1

                    # Increment horizontal angle by azimuth
                    current_height_angle += hrad_res

        # Increment progress bar
        bar.next()

    bar.finish() # Finish progress bar
    return max_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', nargs='?', type=str, required=True, help='Max maps folder absolute path')
    parser.add_argument('--map_size', type=int, default=map_size)
    parser.add_argument('--cell_size', type=float, default=cell_size)
    parser.add_argument('--min_height', type=float, default=min_height)
    parser.add_argument('--max_height', type=float, default=max_height)
    parser.add_argument('--num_slices', type=int, default=num_slices)
    parser.add_argument('--num_planes', type=int, default=num_planes)
    parser.add_argument('--velo_minangle', type=float, default=velo_minangle)
    parser.add_argument('--velo_hres', type=float, default=velo_hres)
    parser.add_argument('--velo_vres', type=float, default=velo_vres)
    parser.add_argument('--velo_height', type=float, default=velo_height)
    args = parser.parse_args()

    # Parse arguments
    map_size=args.map_size
    cell_size=args.cell_size
    min_height=args.min_height
    max_height=args.max_height
    num_slices=args.num_slices
    num_planes=args.num_planes
    velo_minangle=args.velo_minangle
    velo_hres=args.velo_hres
    velo_vres=args.velo_vres
    velo_height=args.velo_height

    path = args.maps
    if not os.path.exists(path):
        os.makedirs(path)

    if path.endswith('/'):
        path = path[:-1]
    
    max_map = get_max_points_map(map_size, cell_size, min_height, max_height, num_slices, num_planes, velo_minangle, velo_hres, velo_vres, velo_height)
    
    print('\nParameters config:')
    print(f'map_size: {map_size}')
    print(f'cell_size: {cell_size}')
    print(f'min_height: {min_height}')
    print(f'max_height: {max_height}')
    print(f'num_slices: {num_slices}')
    print(f'num_planes: {num_planes}')
    print(f'velo_minangle: {velo_minangle}')
    print(f'velo_hres: {velo_hres}')
    print(f'velo_vres: {velo_vres}')
    print(f'velo_height: {velo_height}\n')

    print('Max.: {}; Min.: {}\n'.format(np.max(max_map), np.min(max_map)))

    z_slice = (max_height - min_height) / num_slices
    slices_info = [f"Slice {i}: [{min_height + i * z_slice:.2f}, {min_height + (i + 1) * z_slice:.2f})"
    for i in range(num_slices)]

    print("Height areas for the slices:")
    for info in slices_info:
        print(info)

    cell_str = '{0:.2f}'.format(cell_size)
    velo_height_str = '{0:.2f}'.format(velo_height)

    for n in range(num_slices):
        print('Map created at {}/{}_{}_{}_{}_{}_map.txt'.format(path, map_size, cell_str, num_planes, velo_height_str, 'slice%d' %n))
        np.savetxt('{}/{}_{}_{}_{}_{}_map.txt'.format(path, map_size, cell_str, num_planes, velo_height_str, 'slice%d' %n), max_map[:,:,n]) # fmt='%.2f'


