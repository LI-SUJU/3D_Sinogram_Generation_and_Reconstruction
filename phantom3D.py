import os
import numpy as np
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
import numpy.ma as ma

# 创建输出文件夹
os.makedirs("output", exist_ok=True)

cyl_radius = 2.0
cyl_height = 10.0

stage_params = [
    {'N_voids': 10, 'rmax': 0.4},
    {'N_voids': 10, 'rmax': 0.2},
    {'N_voids': 10, 'rmax': 0.1}
]

N_trial = 1000
min_allowed = 0.01
pitch = 0.04

mesh_color = 'white'
mesh_opacity = 0.3

def random_point_in_cylinder(R, H):
    theta = np.random.uniform(0, 2*np.pi)
    r = np.sqrt(np.random.uniform(0, 1)) * R
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(-H/2, H/2)
    return np.array([x, y, z])

def max_allowed_radius(p, voids, R, H, rmax):
    d_cyl = R - np.linalg.norm(p[:2])
    d_height = (H/2) - abs(p[2])
    boundary_dist = min(d_cyl, d_height)
    if voids:
        dist_to_others = [np.linalg.norm(p - v['center']) - v['radius'] for v in voids]
        d_void = np.min(dist_to_others)
    else:
        d_void = np.inf
    allowed = min(boundary_dist, d_void, rmax)
    for v in voids:
        if np.linalg.norm(p - v['center']) < v['radius']:
            return 0.0
    return allowed

def place_voids_in_stage(voids, N_voids_stage, rmax_stage, N_trial, cyl_radius, cyl_height, min_allowed):
    trial_points = [random_point_in_cylinder(cyl_radius, cyl_height) for _ in range(N_trial)]
    inserted_count = 0
    while inserted_count < N_voids_stage:
        if not trial_points:
            break
        allowed_radii = [max_allowed_radius(p, voids, cyl_radius, cyl_height, rmax_stage) for p in trial_points]
        best_idx = np.argmax(allowed_radii)
        best_r = allowed_radii[best_idx]
        best_point = trial_points[best_idx]
        if best_r < min_allowed:
            break
        voids.append({'center': best_point, 'radius': best_r})
        inserted_count += 1
        new_trial = []
        for p in trial_points:
            if np.linalg.norm(p - best_point) >= best_r:
                new_trial.append(p)
        trial_points = new_trial
        while len(trial_points) < N_trial:
            trial_points.append(random_point_in_cylinder(cyl_radius, cyl_height))
    return inserted_count

voids = []
total_inserted = 0
for stage in stage_params:
    N_v = stage['N_voids']
    r_m = stage['rmax']
    inserted = place_voids_in_stage(voids, N_v, r_m, N_trial, cyl_radius, cyl_height, min_allowed)
    total_inserted += inserted
    print(f"Stage (rmax={r_m}): inserted {inserted} voids so far. Total = {total_inserted}")

print(f"Total number of inserted voids: {total_inserted}")

cylinder = trimesh.creation.cylinder(radius=cyl_radius, height=cyl_height, sections=64)
vox = cylinder.voxelized(pitch=pitch).fill()
points = vox.points
sparse_indices = vox.sparse_indices
mask = np.ones(points.shape[0], dtype=bool)

for v in voids:
    center = v['center']
    r = v['radius']
    dist = np.linalg.norm(points - center, axis=1)
    inside = dist < r
    mask[inside] = False

new_sparse = sparse_indices[mask]
dense = np.zeros(vox.matrix.shape, dtype=bool)
dense[tuple(new_sparse.T)] = True
new_vox = trimesh.voxel.VoxelGrid(dense, transform=vox.transform)
mesh_with_voids = new_vox.marching_cubes

print("Original voxel count:", len(sparse_indices))
print("Remaining voxel count:", len(new_sparse))
print("Number of mesh faces:", len(mesh_with_voids.faces))
print("Number of mesh vertices:", len(mesh_with_voids.vertices))

np.save("phantom_dense.npy", dense)
print("Saved dense volume to phantom_dense.npy")
print("dense shape:", dense.shape)

nz, ny, nx = dense.shape
x_center = nx // 2
y_center = ny // 2
z_center = nz // 2

# 保存各个切片
slice_xy = dense[z_center, :, :]
slice_xz = dense[:, y_center, :]
slice_yz = dense[:, :, x_center]

plt.figure(figsize=(10,10))
plt.imshow(slice_xy, cmap='gray_r', origin='lower')
plt.title("Slice in XY plane (z center)")
plt.axis('off')
plt.savefig("output/slice_xy.png", dpi=600)
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(slice_xz, cmap='gray_r', origin='lower')
plt.title("Slice in XZ plane (y center)")
plt.axis('off')
plt.savefig("output/slice_xz.png", dpi=600)
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(slice_yz, cmap='gray_r', origin='lower')
plt.title("Slice in YZ plane (x center)")
plt.axis('off')
plt.savefig("output/slice_yz.png", dpi=600)
plt.close()

# 保存投影图
proj_xy_sum = np.sum(dense, axis=0)
proj_xz_sum = np.sum(dense, axis=1)
proj_yz_sum = np.sum(dense, axis=2)

plt.figure(figsize=(10,10))
plt.imshow(proj_xy_sum, cmap='inferno', origin='lower')
plt.title("Sum Projection in XY plane (sum over z)")
plt.axis('off')
plt.savefig("output/proj_xy_sum.png", dpi=600)
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
ax.set_facecolor("white")
proj_xy_masked = ma.masked_equal(proj_xy_sum, 0)
cax = ax.imshow(proj_xy_masked, cmap='inferno', origin='lower', interpolation='none')
plt.title("Masked Sum Projection in XY plane")
plt.axis('off')
plt.savefig("output/proj_xy_sum_masked.png", dpi=600, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(proj_xz_sum, cmap='inferno', origin='lower')
plt.title("Sum Projection in XZ plane (sum over y)")
plt.axis('off')
plt.savefig("output/proj_xz_sum.png", dpi=600)
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
ax.set_facecolor("white")
proj_xz_masked = ma.masked_equal(proj_xz_sum, 0)
ax.imshow(proj_xz_masked, cmap='inferno', origin='lower', interpolation='none')
plt.title("Masked Sum Projection in XZ plane")
plt.axis('off')
plt.savefig("output/proj_xz_sum_masked.png", dpi=600, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,10))
plt.imshow(proj_yz_sum, cmap='inferno', origin='lower')
plt.title("Sum Projection in YZ plane (sum over x)")
plt.axis('off')
plt.savefig("output/proj_yz_sum.png", dpi=600)
plt.close()

fig, ax = plt.subplots(figsize=(10,10))
ax.set_facecolor("white")
proj_yz_masked = ma.masked_equal(proj_yz_sum, 0)
ax.imshow(proj_yz_masked, cmap='inferno', origin='lower', interpolation='none')
plt.title("Masked Sum Projection in YZ plane")
plt.axis('off')
plt.savefig("output/proj_yz_sum_masked.png", dpi=600, bbox_inches='tight')
plt.close()

# 保存三维网格模型的渲染结果
faces_np = mesh_with_voids.faces
faces_pv = np.c_[np.full(len(faces_np), 3, dtype=np.int64), faces_np].ravel()
mesh_pv = pv.PolyData(mesh_with_voids.vertices, faces_pv)
plotter = pv.Plotter(off_screen=True)  # 使用 off_screen 模式保存图片
plotter.set_background('black')
light = pv.Light(position=(5, 5, 10), focal_point=(0, 0, 0), color='white')
light.intensity = 1.0
plotter.add_light(light)
plotter.add_mesh(
    mesh_pv,
    color=mesh_color,
    opacity=mesh_opacity,
    smooth_shading=True,
    specular=0.5,
    specular_power=15
)
plotter.camera_position = 'xy'
plotter.show(title="Multi-Stage Foam Phantom")
