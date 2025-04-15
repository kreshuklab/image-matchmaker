# matchmaker
Tool for segmentation-based deformable registration and object matching



## Registration
### Input

- **Fixed image**: 3D instance segmentation in `n5` + resolution 

- **Moving image**: 3D instance segmentation `n5` + resolution 

- **Registration parameters yaml**: parameters of the `matchmaker` pipeline 

Expected image shape: ZYX

### Output
- **Moving image resampled to match fixed image** `n5`
- **QC plots**
- **Files with all transforms**
- **Table of correspondence between instances in moving and fixed instance segmentations**


### Registration steps

0. Create point cloud
1. PCA pre-alignment ---> **rigid transform matrix**
2. Manual input: should the image be flipped?
3. Rigid pre-alignment with Elastix OR with rigid CPD ---> **rigid transform matrix**
4. Coherent point drift
5. Matching points with mixed integer programming
6. Deformable registration with Elastix with distance between keypoints in loss and rigidity penalty ---> **B-spline coefficients**


## Apply registration to other images
### Input

- **Moving image**: 3D image `n5` + resolution
- **Files with all transforms**

### Output
- **Moving image resampled to match fixed image** - only rigid - `n5`
- **Moving image resampled to match fixed image** - deformable - `n5`

Expected image shape: (C)ZYX