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


### Registration steps

**1. PCA pre-alignment**: alignment of fixed and moving image to the PCs \
 `prealignment.py --fixed_path ... --fixed_key ... --moving_path ... --moving_key ... --output_dir ... --mobie_export --dataset_name ...` \
 output: prealigned images + **transformation matrix**

**2. Rigid pre-alignment with Elastix** \
`apply_rigid_elastix.py --fixed_path ... --fixed_key ... --moving_path ... --moving_key ... --output_dir ... --mobie_export --dataset_name ...` \
output: rigid alinged moving image + **rigid transformation matrix**

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