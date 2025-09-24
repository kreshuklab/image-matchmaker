# 💞 Matchmaker
Tool for segmentation-based deformable registration and object matching



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