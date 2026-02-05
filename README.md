# 💞 Matchmaker
Tool for segmentation-based deformable registration and object matching

## How to run

### Create conda environment
```
conda env create -f environment.yml
```


### Run tests

After activating the environment, generate the deformed test data, run the Snakemake workflow, and validate the final results with pytest:

```bash
pytest -s
```

Optional: If you want to hide all warnings during tests, append:

```bash
-p no:warnings
```

This test generates deformed test data, runs the Snakemake workflow, and compares the final outputs against pre-computed reference results.

The reference data will be downloaded automatically from this repo's release if available. If the download fails (e.g. the repository is private), manually download the reference data from [here](https://github.com/kreshuklab/matchmaker/releases/tag/test_data-v0.2) and place it under:

```
examples/data/test_data/
```


### Generate rigid and elastic deformed test data
```
python examples/deform_test_data.py
```


### 3 ways of interaction with the library

- Running the full pipeline using workflow manager and a config file to set up registration parameters
```
snakemake -s workflows/register.smk --cores 16
```
- Running separate scripts
```
prealignment.py --fixed_path ... --fixed_key ... --moving_path ... --moving_key ... --output_dir ... --mobie_export --dataset_name ...
```

- Importing individual functions from matchmaker:
```
import matchmaker as mm

...

mm.n5-utils.read_volume(...)
```



## Registration
### Input

- **Fixed image**: 3D instance segmentation in `n5` + resolution 
- **Moving image**: 3D instance segmentation `n5` + resolution 


Expected image shape: ZYX

### Output
- **Moving image resampled to match fixed image** `n5`
- **QC plots**
- **Files with all transforms**
- **Table of correspondence between instances in moving and fixed instance segmentations**
- **Logging file: `registration.log`**
- **Optional: Mobie project saved at `{output_dir}/mobie_project/`**


### Registration steps

**1. PCA pre-alignment**: alignment of fixed and moving image to the PCs \
 `prealignment.py --fixed_path ... --fixed_key ... --moving_path ... --moving_key ... --output_dir ... --mobie_export --dataset_name ...` \

Outputs: 
- prealigned images: `{file_name}_prealigned.n5`
- transformation matrixes
    - `{file_name}_fixed_T_prealignment.txt`
    - `{file_name}_moving_T_prealignment.txt` (maybe final one is `moving_T_prealignment.txt`, couldn't figure this out)
- plots:
    - slice per dimension before pre-alignment: `{file_name}_fixed.png`, `{file_name}_moving.png`
    - slice per dimension after pre-alignment: `{file_name}_fixed_prealigned.png`, `{file_name}_moving_prealigned.png`
    - overlay of slice per dimension after pre-alignment: `overlay_prealignment.png`
    - intensity profiles per axis and volume: `fixed_intensity_profile_{axis}.png`, `moving_intensity_profile_{axis}.png`

**2. Rigid pre-alignment with Elastix** \
`apply_rigid_elastix.py --fixed_path ... --fixed_key ... --moving_path ... --moving_key ... --output_dir ... --mobie_export --dataset_name ...` \

Outputs: 
- rigid alinged moving image: `{file_name}_rigid_aligned.n5`
- rigid transformation matrix (Elastix outputs): `result.0.mhd`, `result.0.raw`, `TransformParameters.0.txt`
- logging file: `elastix_log_rigid.log`
- plots:
    - `intersample_segm_overlay_before_alignment.png`
    - `intersample_segm_rigid_alignment_semantic.png`


**3. Coherent point drift**

**4. Matching points with mixed integer programming**

**5. Deformable registration with Elastix** \
with distance between keypoints in loss and rigidity penalty ---> **B-spline coefficients**




## Apply registration to other images
### Input

- **Moving image**: 3D image `n5` + resolution
- **Files with all transforms**

### Output
- **Moving image resampled to match fixed image** - only rigid - `n5`
- **Moving image resampled to match fixed image** - deformable - `n5`

Expected image shape: (C)ZYX