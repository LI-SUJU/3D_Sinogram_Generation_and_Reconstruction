
# 3D Sinogram Generation and Reconstruction

This project simulates a 3D foam phantom, generates sinograms under various experimental settings, and reconstructs the phantom using FBP and SIRT algorithms with the ASTRA Toolbox.

## Setup and Dependencies

Before running the code, make sure the required environment is set up:

```bash
pip install numpy trimesh pyvista matplotlib
conda install -c astra-toolbox astra-toolbox --yes
```

## Usage Instructions

### Step 1: Generate Phantom

Run the following script to create the 3D phantom and save the volume to `phantom_dense.npy`:

```bash
python phantom3D.py
```

This will also generate 2D slice and projection images in the `output/` folder.

### Step 2: Generate Sinograms

Run the sinogram generation script:

```bash
python sinogram3D.py
```

You can control the following parameters by editing the script:

- `n_proj` — Number of projection angles (e.g., 30, 90, 180)
- `add_noise` — Whether to add Gaussian noise (True or False)
- `noise_std` — Standard deviation of the noise (e.g., 0.05)

The resulting sinogram files and their visualizations will be saved in the `output/` directory.

### Step 3: Reconstruct the Volume

Run the reconstruction script:

```bash
python reconstruct3D.py
```

You can modify the following parameters in the script:

- `method` — Reconstruction algorithm ('FBP' or 'SIRT')
- `n_iter` — Number of iterations for SIRT (ignored if using FBP)

Reconstruction results and visualization images will be saved in the `output/` folder.

## File Overview

- `phantom3D.py`: Generates the 3D phantom volume and projections.
- `sinogram3D.py`: Generates sinograms based on configurable settings.
- `reconstruct3D.py`: Performs 2D slice-by-slice reconstruction using FBP or SIRT.
- `output/`: Contains all generated files including `.npy` arrays and `.png` visualizations.

## Notes

- This project uses 2D reconstruction (slice by slice) to simulate 3D reconstruction due to ASTRA 3D limitations on some platforms.

```
