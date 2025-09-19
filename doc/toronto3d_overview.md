# KPConv-PyTorch + Toronto3D: Deep-dive Overview

This document summarizes how KPConv-PyTorch is structured, how the Toronto3D dataset is wired into the pipeline, and what you need to prepare and run training/validation on Windows.

## Big picture

- KPConv is a point-based convolution operating on local neighborhoods around query points via fixed or deformable kernel points.
- The repository defines generic building blocks (KPConv, residual blocks, pooling/upsampling) and dataset-agnostic input builders that compute multi-scale neighborhoods via C++ extensions (grid subsampling and radius neighbors).
- Toronto3D is integrated as a cloud segmentation dataset (scene-level segmentation with sliding spheres). The pipeline samples spheres by potential (for coverage), builds multi-scale neighborhoods per sphere, runs a U-Net style KPFCNN, and aggregates probabilistic votes back to full clouds for metrics.

## Core modules and flow

- models/blocks.py
  - KPConv: core operator. Optional deformable and modulated variants. Computes kernel influences over neighbors and applies learnable weights.
  - SimpleBlock and ResnetBottleneckBlock: wrap KPConv with MLP, BN, ReLU; support strided/pooling variants. Pool uses max over indices.
  - MaxPoolBlock, NearestUpsampleBlock, GlobalAverageBlock: graph topology ops for encoder/decoder and classification heads.
- models/architectures.py
  - KPCNN: classification head with stacked blocks -> global MLP -> logits
  - KPFCNN: encoder-decoder (U-Net-like) for segmentation. Tracks skip dims, constructs decoder with concatenations at upsample stages; head projects to C classes.
  - p2p_fitting_regularizer: regularizes deformable kernels (point-to-point fit and repulsion between kernel points).
- kernels/kernel_points.py
  - Generates fixed kernel point dispositions (Lloyd or optimization). Saves to kernels/dispositions/k_XXX_center_3D.ply; applies random rotation and small noise each load.
- datasets/common.py
  - PointCloudDataset: base class with augmentation, batching, and input builders:
    - grid_subsampling and batch_neighbors call C++ ops for performance.
    - classification_inputs / segmentation_inputs compute per-layer: points, neighbors, pools, upsamples, and stack lengths.
  - C++ wrappers are required: cpp_wrappers/cpp_subsampling and cpp_wrappers/cpp_neighbors build Python extensions grid_subsampling and radius_neighbors.
- utils/trainer.py
  - ModelTrainer: SGD with separate LR for deform offsets; training loop logs, checkpoints, validation per dataset type.
  - cloud_segmentation_validation: aggregates softmax votes across spheres into whole-cloud predictions; computes IoU from confusion matrices.
- utils/tester.py
  - ModelTester: performs evaluation/voting and optional reprojection/PLY exports.
- utils/config.py
  - Config: central hyperparameters; dataset configs subclass and override specifics.

## Toronto3D integration

- datasets/Toronto3D.py defines Toronto3DDataset (cloud_segmentation) and Toronto3DSampler.
- Expected on-disk structure under the repo root:
  - Data/Toronto3D/
    - original_ply/
      - L001.ply, L002.ply, L003.ply, L004.ply (raw MLS clouds with fields: x, y, z, red, green, blue, scalar_Intensity, scalar_Label)
    - train/ (auto-generated PLYs re-centered by UTM_OFFSET and copied fields)
    - input_0.080/ (auto-generated KDTree, subsampled PLYs, projections; name depends on first_subsampling_dl)
- Label map (9 classes, with label 0 ignored):
  0 Unclassified (ignored)
  1 Road
  2 Road_markings
  3 Natural
  4 Building
  5 Utility_line
  6 Pole
  7 Car
  8 Fence
- Splits inside the dataset class:
  - validation_split = 1 (L002 is val by default). Training uses L001, L003, L004; validation uses L002.
- Features and inputs:
  - in_features_dim=4 (1 for bias + RGB) or 5 (bias + RGB + height Z). In current Toronto3DConfig it sets in_features_dim=4; dataset will assemble features accordingly. Height feature is constructed as Z residual + center Z when in_features_dim=5.
- Sampling and potentials:
  - use_potentials=True by default uses coarse KDTree over subsampled points to select sphere centers with Tukey update to ensure even coverage over epochs.
  - in_radius=3.0 m default; batch_num=4 spheres per batch; calibration finds batch_limit (#points per batch) and neighborhood_limits per layer such that most neighborhoods are untouched.
- Data preparation inside dataset:
  - prepare_Toronto3D_ply: reads original_ply/L00X.ply, subtracts UTM_OFFSET [627285, 4841948, 0], writes to Data/Toronto3D/train/L00X.ply with fields [x,y,z, rgb, scalar_Intensity, scalar_Label].
  - load_subsampled_clouds: for each cloud, if missing, subsamples to dl=first_subsampling_dl via grid_subsampling, builds KDTree, stores to Data/Toronto3D/input_XXX.
  - For validation, prepares test_proj (mapping from original points to subsampled KDTree) and validation_labels.

## Training script (train_Toronto3D.py)

- Toronto3DConfig: tailored hyperparameters
  - Architecture: 4 down-sampling stages (resnetb_strided) then upsample stages with unary heads between.
  - in_radius=3.0, first_subsampling_dl=0.08, conv_radius=2.5, deform_radius=5.0, KP_extent=1.0, KP_influence=linear, aggregation_mode=closest
  - first_features_dim=128, in_features_dim=4, BN momentum 0.02
  - Training: max_epoch=400, lr=1e-2 with slow decay, momentum 0.98, batch_num=4, epoch_steps=500, validation_size=50.
- Pipeline:
  1) Instantiate Toronto3DDataset twice (training, validation) with use_potentials=True
  2) Create Samplers and DataLoaders (num_workers=config.input_threads)
  3) Calibrate samplers (sets batch_limit and neighborhood_limits and caches them in Data/Toronto3D/*.pkl)
  4) Build KPFCNN(net) passing label_values and ignored_labels for loss mapping
  5) Train with ModelTrainer; checkpoints go to results/Log_YYYY-mm-dd_HH-MM-SS by default (or override saving_path).

## C++ extensions (Windows notes)

- grid_subsampling and radius_neighbors are Python extensions built from C++ sources.
- Build scripts: cpp_wrappers/cpp_subsampling/build.bat and cpp_wrappers/cpp_neighbors/build.bat simply run `py setup.py build_ext --inplace`.
- Prereqs: Visual C++ Build Tools compatible with your Python; numpy headers; a MSVC that accepts C++11. The provided extra_compile_args use GCC-style flags; on MSVC these are ignored or may need adaptation, but prior users have built on Windows 10 successfully. If build errors mention `-std=c++11` or `-D_GLIBCXX_USE_CXX11_ABI`, remove or guard those flags for MSVC.

## File fields and numeric stability

- The dataset stores UTM in meters with large magnitudes; subtracting UTM_OFFSET in prepare_Toronto3D_ply ensures FP32 precision is sufficient for grid_subsampling and neighbors. This mirrors the Toronto-3D README tip. If you bring different tiles, ensure you offset coordinates similarly.
- PLY schema expected by dataset code:
  - Raw input (original_ply): x, y, z, red, green, blue, scalar_Intensity, scalar_Label.
  - Subsampled PLYs in input_XXX: same fields but colors are normalized to [0,1] float; labels are ints.

## What the model learns and regularization

- KPConv weights per kernel point W(K, Fin, Fout) and optionally offsets for deformable kernels.
- p2p_fitting_regularizer encourages deformed kernel points to lie near input samples and repels kernel points from each other.
- Loss: CrossEntropy on valid classes, ignore_index=-1 for ignored class 0 mapping.

## Aggregation and metrics

- During validation, probabilities for each subsampled point accumulate with exponential smoothing (val_smooth=0.95). Periodically projected to original points using test_proj indexing.
- Confusion matrices computed on full validation clouds (after projection) and IoUs per class; mIoU reported.

## Dataset splits and class encoding

- By default: train on L001, L003, L004; validate on L002.
- Class IDs follow Mavericks_classes_9.txt with 0 ignored in loss.
- Colors.xml provides CloudCompare color scale for visualization; has no impact on training.

## Prepare and run on Windows (PowerShell)

1) Create and activate a virtual environment (Python 3.10 recommended):

```powershell
# From KPConv-PyTorch directory
py -3.10 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Install Python packages (match your CUDA/torch version):

```powershell
# Example for CUDA 12.1 (adjust if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn pyyaml matplotlib tqdm
```

3) Build native extensions:

```powershell
# Requires MSVC Build Tools and matching VS C++ toolchain in PATH
cd cpp_wrappers\cpp_subsampling
py setup.py build_ext --inplace
cd ..\cpp_neighbors
py setup.py build_ext --inplace
cd ..\..
```

- If compilation fails on flags, edit setup.py in each wrapper and remove `'-std=c++11'` and `'-D_GLIBCXX_USE_CXX11_ABI=0'` for MSVC. Numpy include dirs are automatically injected.

4) Prepare dataset folder layout under repo root:

```text
KPConv-PyTorch\
  Data\Toronto3D\original_ply\L001.ply, L002.ply, L003.ply, L004.ply
```

- You already have copies under Toronto_3D/Toronto_3D; copy them into Data/Toronto3D/original_ply.

5) First run will auto-generate train PLYs and subsampled KDTree caches; then run calibration, then training:

```powershell
python train_Toronto3D.py
```

- Outputs go to results/Log_...; checkpoints under checkpoints/; training.txt logs metrics each step.

## Quick smoke test (dataset + loaders)

- Create a small script (verify_toronto3d.py) that instantiates the dataset with a tiny in_radius and first_subsampling_dl, builds loaders, runs one batch through KPFCNN, and prints shapes. See the companion steps the project’s TODO tracks.

## Common pitfalls and tips

- Ensure CUDA and torch build match your GPU; if CPU-only, set GPU_ID to an invalid index or leave CUDA invisible.
- Large memory: in_radius and first_subsampling_dl control batch memory; if OOM, increase first_subsampling_dl (e.g., 0.12) and/or reduce in_radius (e.g., 2.0) and/or lower batch_num.
- Always run samplers.calibration after changing those parameters to recompute batch_limit and neighborhood_limits.
- Colors in subsampled PLYs are normalized; if you want intensity or height, set in_features_dim accordingly.

## What to modify for experiments

- Change architecture in Toronto3DConfig.architecture; use deformable variants ('resnetb_deformable', etc.) or change aggregation_mode ('sum') and num_kernel_points.
- Adjust augmentations: augment_color probability, rotation mode ('vertical' vs 'all'), noise.
- Loss weighting: set class_w in Config if you wish to reweight rare classes (e.g., Road_markings, Fence).

## References

- KPConv (ICCV 2019): https://arxiv.org/abs/1904.08889
- Toronto-3D dataset: CVPRW 2020 paper and README included under Toronto_3D/Toronto_3D.
