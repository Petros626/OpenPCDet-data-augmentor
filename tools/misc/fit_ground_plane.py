import numpy as np

def fit_ground_plane_ransac_(points, n_iterations=1000, inlier_threshold=0.05):
    best_plane = None
    max_inliers = 0

    for _ in range(n_iterations):
        # Randomly select three non-collinear points to form a plane
        indices = np.random.choice(points.shape[0], 3, replace=False)
        sample_points = points[indices]

        # Calculate the coefficients of the plane equation (A, B, C, D)
        p1, p2, p3 = sample_points
        normal_vector = np.cross(p2 - p1, p3 - p1)
        A, B, C = normal_vector
        D = -np.dot(normal_vector, p1)

        # Calculate the distance of all points to the estimated plane
        distances = np.abs(np.dot(points, np.array([A, B, C])) + D) / np.linalg.norm([A, B, C])

        # Count the number of inliers
        inliers = np.sum(distances < inlier_threshold)

        # Update the best plane if this iteration has more inliers
        if inliers > max_inliers:
            max_inliers = inliers
            best_plane = (A, B, C, D)

    return best_plane


def plane_coefficients_to_euler(coefficients):
    A, B, C, _ = coefficients

    # Extract the normal vector from plane coefficients
    normal_vector = np.array([A, B, C])

    # Normalize the normal vector to get unit normal
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate roll, pitch, and yaw angles
    roll = np.arctan2(normal_vector[1], normal_vector[2])  # Rotation around x-axis
    pitch = np.arctan2(-normal_vector[0], np.sqrt(normal_vector[1]**2 + normal_vector[2]**2))  # Rotation around y-axis
    yaw = 0  # We assume yaw to be 0, as the ground plane does not have a rotation around the z-axis

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def fit_ground_plane(points, verbose=True):
    plane_coefficients = fit_ground_plane_ransac_(points[:, :3])
    if verbose: print("Best Plane Coefficients:", plane_coefficients)

    # Assuming you have the plane coefficients in the 'plane_coefficients' variable
    roll_angle, pitch_angle, yaw_angle = plane_coefficients_to_euler(plane_coefficients)
    z_offset = -plane_coefficients[3]/plane_coefficients[2]
    if verbose: 
        print("Roll Angle (x-axis):", roll_angle)
        print("Pitch Angle (y-axis):", pitch_angle)
        print("z-offset of the plane:", z_offset)

    return roll_angle, pitch_angle, z_offset