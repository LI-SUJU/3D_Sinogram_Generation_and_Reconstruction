import numpy as np
import astra
import matplotlib.pyplot as plt
import os


def reconstruct_3d_from_2d(
        sino_path="output_3d/full3D_sino_180.npy",
        n_proj=180,
        nx=101,
        ny=101,
        method='FBP',  # 目前支持 'FBP'，你也可以扩展为 'SIRT'
        n_iter=100,  # SIRT 迭代次数（如果采用 SIRT）
        save_prefix="recon3d"
):
    # Step 1: 加载完整 3D sinogram，形状应为 (n_proj, det, nz)
    full_sino = np.load(sino_path)
    print("[INFO] Loaded full 3D sinogram with shape:", full_sino.shape)
    # 假设 full_sino.shape = (n_proj, det, nz)
    n_proj_loaded, det, nz = full_sino.shape
    if n_proj_loaded != n_proj:
        print("[WARN] n_proj in file ({}) differs from specified ({})".format(n_proj_loaded, n_proj))

    # Step 2: 准备存储重建结果，最终 3D 体积 shape = (nz, ny, nx)
    recon_volume = np.zeros((nz, ny, nx), dtype=np.float32)

    # Step 3: 为 2D 重建定义投影几何和体积几何（与生成 sinogram 时保持一致）
    # 在 2D 中，投影几何为平行光束，探测器长度为 nx
    proj_geom_2d = astra.create_proj_geom('parallel', 1.0, nx, np.linspace(0, np.pi, n_proj, endpoint=False))
    vol_geom_2d = astra.create_vol_geom(nx, ny)

    # Step 4: 对每个 z 层逐层重建
    print("[INFO] Starting 2D reconstruction for each slice...")
    for iz in range(nz):
        # 提取第 iz 层 sinogram，形状 = (n_proj, det)
        sino_2d = full_sino[:, :, iz]

        # 创建 ASTRA 数据对象
        sino_id = astra.data2d.create('-sino', proj_geom_2d, sino_2d)
        rec_id = astra.data2d.create('-vol', vol_geom_2d)

        if method.upper() == 'FBP':
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)
        elif method.upper() == 'SIRT':
            cfg = astra.astra_dict('SIRT_CUDA')
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, n_iter)
            astra.algorithm.delete(alg_id)
        else:
            raise ValueError("Unknown reconstruction method: " + method)

        # 取回重建结果
        recon_slice = astra.data2d.get(rec_id)
        recon_volume[iz] = recon_slice

        # 清理当前 slice 数据
        astra.data2d.delete([sino_id, rec_id])

        if (iz + 1) % 50 == 0 or iz == nz - 1:
            print(f"[INFO] Reconstructed slice {iz + 1}/{nz}")

    # Step 5: 保存 3D 重建体积及部分可视化
    os.makedirs("output_3d", exist_ok=True)
    out_npy = f"output_3d/{save_prefix}_{method}_{n_proj}.npy"
    np.save(out_npy, recon_volume)
    print("[INFO] 3D reconstruction saved to", out_npy)

    # 可视化：显示重建体积的中心切片（沿 z 轴、y 轴、x 轴）
    mid_z = nz // 2
    mid_y = ny // 2
    mid_x = nx // 2

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(recon_volume[mid_z, :, :], cmap="gray")
    plt.title(f"Z-slice (z={mid_z})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recon_volume[:, mid_y, :], cmap="gray")
    plt.title(f"Y-slice (y={mid_y})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(recon_volume[:, :, mid_x], cmap="gray")
    plt.title(f"X-slice (x={mid_x})")
    plt.axis('off')

    plt.tight_layout()
    out_png = f"output_3d/{save_prefix}_{method}_{n_proj}_vis.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("[INFO] Visualization saved to", out_png)

    return recon_volume


if __name__ == "__main__":
    # 你可以选择 FBP 或 SIRT 方法（此处默认 FBP）
    recon_volume = reconstruct_3d_from_2d(
        sino_path="output_3d/full3D_sino_180.npy",
        n_proj=180,
        nx=101,
        ny=101,
        method='FBP',  # 或改为 'SIRT'
        n_iter=100,
        save_prefix="recon3d"
    )
