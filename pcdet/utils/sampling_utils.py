import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn import linear_model

def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)

    phi[phi_ < 0] = 2*np.pi - phi[phi_ < 0]
    phi[phi == 2*np.pi] = 0

    return theta, phi

def beam_label(theta, beam, method='kmeans'):
    if method == 'kmeans++':
        estimator=KMeans(n_clusters=beam, init='k-means++', n_init="auto")
    elif method == 'kmeans':
        estimator=KMeans(n_clusters=beam, init='random', n_init='auto')
    elif method == 'dbscan':
        estimator=DBSCAN(eps=0.1, min_samples=10)
   
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_

    #sort beam index based on the mean theta of each beam,
    #so that the beam index is consistent across different scans
    mean_theta = np.zeros((beam))
    for i in range(beam):
        mean_theta[i] = np.mean(theta[label == i])
    sorted_inds = np.argsort(mean_theta)
    sorted_labels = np.ones_like(label) * (-1)
    for i in range(beam):
        sorted_labels[label == sorted_inds[i]] = i  

    return sorted_labels 

def beam_label_ransac(polar_image, num_beams, inlier_threshold=0.1):
    # polar_image: theta, phi
    theta = polar_image[:, 0]
    phi = polar_image[:, 1]
    theta = theta.reshape(-1, 1)
    phi = phi.reshape(-1, 1)

    label = np.zeros((theta.shape[0]), dtype=np.int32)
    
    unassigned_theta = theta.copy()
    unassigned_phi = phi.copy()

    remaining_inds = np.arange(theta.shape[0])
    for i in range(num_beams):
        ransac = linear_model.RANSACRegressor(residual_threshold=inlier_threshold)
        ransac.fit(unassigned_theta, unassigned_phi)
        inlier_mask = ransac.inlier_mask_

        #track assigned points
        assigned_inds = remaining_inds[inlier_mask]
        label[assigned_inds] = i

        #remove assigned points
        remaining_inds = remaining_inds[~inlier_mask]
        unassigned_theta = unassigned_theta[~inlier_mask]
        unassigned_phi = unassigned_phi[~inlier_mask]
    
    #sort beam labels according to the mean theta of each beam
    mean_theta = np.zeros((num_beams))
    for i in range(num_beams):
        mean_theta[i] = np.mean(theta[label == i])
    sorted_inds = np.argsort(mean_theta)
    for i in range(num_beams):
        label[label == sorted_inds[i]] = i

    return label

def generate_mask(phi, beam, label, idxs, beam_ratio, bin_ratio):
    mask = np.zeros((phi.shape[0])).astype(np.bool)

    for i in range(0, beam, beam_ratio):
        phi_i = phi[label == idxs[i]]
        idxs_phi = np.argsort(phi_i)
        mask_i = (label == idxs[i])
        mask_temp = np.zeros((phi_i.shape[0])).astype(np.bool)
        mask_temp[idxs_phi[::bin_ratio]] = True
        mask[mask_i] = mask_temp

    return mask