bio_matchmaker documentation
============================

**Matchmaker** is a framework that leverages segmentation masks to achieve
alignment between two volumetric datasets of different modalities.

Matchmaker is designed to optimize a transformation that maps a moving
image/volume to align with a fixed image/volume.

- Moving Volume: The source dataset that is spatially transformed (warped)
  to match the target.
- Fixed Volume: The stationary reference dataset that defines the target
  coordinate space for the registration.

.. figure:: _static/images/workflow.png
   :width: 90%
   :align: center

   Overview of the Matchmaker registration workflow.

The workflow begins by taking two segmentation masks from different modalities
(e.g., EM and LM) as the primary inputs. Then these masks enter a sequential
alignment pipeline to progressively refine the spatial correspondence between
the volumes.

The pipeline consists of following steps:

1. **Pre-alignment (SVD):** The masks undergo an initial global alignment using
   Singular Value Decomposition (SVD). This provides a coarse starting
   point by aligning the centroids and principal axes of the segmented
   structures.
2. **Rigid Registration (Elastix):** Following pre-alignment, a rigid
   transformation is performed using the *Elastix* toolbox to account for
   basic rotation and translation differences.
3. **Coherent Point Drift (CPD) Registration:** To further refine the alignment,
   CPD registration is applied to the point clouds derived from the rigid
   registration results using *probreg*, allowing for more nuanced local
   adjustments.
4. **Feature Matching:** The workflow then identifies specific correspondences
   (e.g., matching individual nuclei) between the two volumes using *cvxpy*
   to establish a set of definitive landmarks shared by both datasets.
5. **B-Spline Registration (Elastix):** The final deformable alignment is
   performed using a B-spline transformation.

   -  *Note:* Crucially, the B-Spline Registration step utilizes the original
      segmentation masks and the established matching landmarks as direct inputs,
      rather than relying on the intermediate results from the CPD stage

 **Final Transformation Application**
 The result of the B-spline registration is a sequential transformation comprising
 three refined stages: rigid, rough B-spline, and fine B-spline. This sequence can
 be applied directly to the target channels of the original volumes (e.g., raw EM
 or fluorescence LM) to bring them into a unified coordinate space.

----

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   config_ref
   outputs
   example_data
