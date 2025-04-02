import numpy as np
import astra
import matplotlib.pyplot as plt
import os


def generate_full_3d_sinogram(
        dense_path="phantom_dense.npy",
        n_proj=180,
        add_noise=False,
        noise_std=0.05,
        save_prefix="full3D_sino"
):
    # Step 1: 加载 dense 数据（假设原始 shape 为 (y, x, z)）
    dense = np.load(dense_path)
    print("[INFO] Loaded dense shape:", dense.shape)

    # Step 2: 转置数据为 (z, y, x)
    # 假设原始数据排列为 (y, x, z)，转置后变成 (z, y, x)
    dense = np.transpose(dense, (2, 0, 1))
    nz, ny, nx = dense.shape
    print("[INFO] Transposed volume shape (z,y,x):", (nz, ny, nx))

    # Step 3: 定义投影角度
    angles = np.linspace(0, np.pi, n_proj, endpoint=False).astype(np.float32)

    # Step 4: 对每一层（z方向）生成 2D sinogram
    sino_list = []
    for i in range(nz):
        slice_2d = dense[i].astype(np.float32)  # 每层 shape: (y, x)

        # 创建 2D 投影几何（平行光束）
        proj_geom = astra.create_proj_geom('parallel', 1.0, nx, angles)
        vol_geom = astra.create_vol_geom(nx, ny)

        # 创建 ASTRA 数据对象
        sino_id = astra.data2d.create('-sino', proj_geom)
        vol_id = astra.data2d.create('-vol', vol_geom, slice_2d)
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)

        # 配置前向投影算法（使用 GPU 版 FP）
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectorId'] = proj_id
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # 提取 sinogram 数据（形状应为 (n_proj, det)）
        sino_data = astra.data2d.get(sino_id)

        # 清理资源
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([sino_id, vol_id])
        astra.projector.delete(proj_id)

        sino_list.append(sino_data)

        if (i + 1) % 50 == 0 or i == nz - 1:
            print(f"[INFO] Processed slice {i + 1}/{nz}")

    # Step 5: 将所有 2D sinogram 沿第三个维度叠加
    # 最终 full_sino shape = (n_proj, det, nz)
    full_sino = np.stack(sino_list, axis=-1)
    print("[INFO] Full 3D sinogram shape:", full_sino.shape)

    # Step 6: 添加高斯噪声（可选）
    if add_noise:
        print("[INFO] Adding Gaussian noise with std =", noise_std)
        full_sino += np.random.normal(0, noise_std, size=full_sino.shape).astype(np.float32)

    # Step 7: 保存 sinogram 数据和示意图
    os.makedirs("output_3d", exist_ok=True)
    out_npy = f"output_3d/{save_prefix}_{n_proj}.npy"
    np.save(out_npy, full_sino)
    print("[INFO] Full 3D sinogram saved to", out_npy)

    # 可视化：选取中间投影角度，显示各层数据（横轴：slice index）
    mid_proj = n_proj // 2
    plt.figure(figsize=(8, 6))
    plt.imshow(full_sino[mid_proj, :, :], cmap="gray", aspect="auto")
    plt.title(f"Full 3D Sinogram at projection {mid_proj}")
    plt.xlabel("Slice index (z)")
    plt.ylabel("Detector pixel")
    plt.colorbar()
    plt.tight_layout()
    out_png = f"output_3d/{save_prefix}_{n_proj}_vis.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[INFO] Visualization saved to", out_png)


if __name__ == "__main__":
    generate_full_3d_sinogram(
        dense_path="phantom_dense.npy",
        n_proj=180,
        add_noise=True,
        noise_std=0.05,
        save_prefix="full3D_sino"
    )
