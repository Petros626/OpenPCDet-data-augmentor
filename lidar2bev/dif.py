import numpy as np
import matplotlib.pyplot as plt

def load_kitti_bin(bin_path: str) -> np.ndarray:
    """Loads a KITTI .bin Point Cloud and returns only x, y, z coordinates."""
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # [x, y, z]

def original_compute_density(point_cloud: np.ndarray, cell_size: float = 0.1) -> tuple:
    """Original implementation using np.add.at"""
    x, y = point_cloud[:, 0], point_cloud[:, 1]
    
    x_min, y_min = np.min(x), np.min(y)
    x_max, y_max = np.max(x), np.max(y)
    
    x_indices = ((x - x_min) / cell_size).astype(int)
    y_indices = ((y - y_min) / cell_size).astype(int)
    
    grid_x = int((x_max - x_min) / cell_size) + 1
    grid_y = int((y_max - y_min) / cell_size) + 1
    
    density_map = np.zeros((grid_y, grid_x), dtype=np.int32)
    np.add.at(density_map, (y_indices, x_indices), 1)
    
    return density_map, grid_x * grid_y

def histogram_compute_density(points: np.ndarray, cell_size: float = 0.1) -> tuple:
    """New implementation using np.histogram2d"""
    x = points[:, 0]
    y = points[:, 1]
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    nx = int(np.ceil((x_max - x_min) / cell_size))
    ny = int(np.ceil((y_max - y_min) / cell_size))
    
    grid, _, _ = np.histogram2d(
        x, y,
        bins=[nx, ny],
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    return grid, nx * ny

# Load KITTI data
bin_file = "/home/rlab10/Downloads/training/velodyne/000000.bin"
point_cloud = load_kitti_bin(bin_file)

# Compute densities using both methods
original_density, original_cells = original_compute_density(point_cloud)
histogram_density, histogram_cells = histogram_compute_density(point_cloud)

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot original points (downsample for visualization)
downsample_rate = 1  # Show every point
x = point_cloud[::downsample_rate, 0]
y = point_cloud[::downsample_rate, 1]
scatter = ax1.scatter(x, y, c='blue', alpha=0.1, s=1)
if downsample_rate == 1:
    ax1.set_title('Original Points')
else:
    ax1.set_title('Original Points (downsampled)')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.grid(True)

# Plot original implementation result
im1 = ax2.imshow(original_density, origin='lower', cmap='viridis')
ax2.set_title(f'Original Implementation\nMax Density: {np.max(original_density)}')
plt.colorbar(im1, ax=ax2)
ax2.set_xlabel('Grid Cells (0.1m)')
ax2.set_ylabel('Grid Cells (0.1m)')

# Plot histogram implementation result
im2 = ax3.imshow(histogram_density, origin='lower', cmap='viridis')
ax3.set_title(f'Histogram Implementation\nMax Density: {np.max(histogram_density)}')
plt.colorbar(im2, ax=ax3)
ax3.set_xlabel('Grid Cells (0.1m)')
ax3.set_ylabel('Grid Cells (0.1m)')

# Add difference statistics
print(f"\nDifferences between implementations:")
print(f"Grid shapes - Original: {original_density.shape}, Histogram: {histogram_density.shape}")
print(f"Total cells - Original: {original_cells}, Histogram: {histogram_cells}")
print(f"Maximum density - Original: {np.max(original_density)}, Histogram: {np.max(histogram_density)}")
print(f"Average density - Original: {np.mean(original_density):.2f}, Histogram: {np.mean(histogram_density):.2f}")

# Calculate difference matrix (only where shapes match)
min_rows = min(original_density.shape[0], histogram_density.shape[0])
min_cols = min(original_density.shape[1], histogram_density.shape[1])
diff = original_density[:min_rows, :min_cols] - histogram_density[:min_rows, :min_cols]
print(f"Maximum absolute difference: {np.max(np.abs(diff))}")
print(f"Average absolute difference: {np.mean(np.abs(diff)):.2f}")

# Add zoomed region of high density
# Find region with highest density
max_pos = np.unravel_index(np.argmax(original_density), original_density.shape)
zoom_size = 50  # Show 50x50 cells around maximum
y_start = max(0, max_pos[0] - zoom_size//2)
y_end = min(original_density.shape[0], max_pos[0] + zoom_size//2)
x_start = max(0, max_pos[1] - zoom_size//2)
x_end = min(original_density.shape[1], max_pos[1] + zoom_size//2)

fig2, (ax4) = plt.subplots(1, 1, figsize=(5, 5))

# Zoom original implementation
im3 = ax4.imshow(original_density[y_start:y_end, x_start:x_end], origin='lower', cmap='viridis')
ax4.set_title(f'Original Implementation (Zoomed)\nMax Density Region')
plt.colorbar(im3, ax=ax4)

# Zoom histogram implementation
# im4 = ax5.imshow(histogram_density[y_start:y_end, x_start:x_end], origin='lower', cmap='viridis')
# ax5.set_title(f'Histogram Implementation (Zoomed)\nMax Density Region')
# plt.colorbar(im4, ax=ax5)

plt.tight_layout()
plt.show()