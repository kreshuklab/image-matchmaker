API Reference
=============

This page documents the most useful functions for calling image-matchmaker from
Python. For usage patterns and examples, see the Python API section of the
:doc:`Usage <usage>` page.

The helpers below are re-exported from ``image_matchmaker.utils``; the pipeline-stage
functions live in their respective modules.

I/O
---

.. autofunction:: image_matchmaker.utils.read_volume
.. autofunction:: image_matchmaker.utils.write_volume
.. autofunction:: image_matchmaker.utils.get_attrs
.. autofunction:: image_matchmaker.utils.set_attrs
.. autofunction:: image_matchmaker.utils.load_config
.. autofunction:: image_matchmaker.utils.download_file

Transforms
----------

.. autofunction:: image_matchmaker.utils.rotate_img
.. autofunction:: image_matchmaker.utils.resample_volume
.. autofunction:: image_matchmaker.utils.get_transformation_matrix
.. autofunction:: image_matchmaker.utils.write_transform_dict
.. autofunction:: image_matchmaker.utils.read_transform_dict

Point clouds
------------

.. autofunction:: image_matchmaker.utils.extract_centroids
.. autofunction:: image_matchmaker.utils.create_pcd
.. autofunction:: image_matchmaker.utils.sparse_ilp_matching
.. autofunction:: image_matchmaker.utils.hungarian_matching
.. autofunction:: image_matchmaker.utils.sinkhorn_matching

Visualization
-------------

.. autofunction:: image_matchmaker.utils.plot_three_slices
.. autofunction:: image_matchmaker.utils.plot_overlay
.. autofunction:: image_matchmaker.utils.overlay_pcds
.. autofunction:: image_matchmaker.utils.plot_pcd_overlay_panels
.. autofunction:: image_matchmaker.utils.visualize_displacement_field
.. autofunction:: image_matchmaker.utils.plot_displacement_field_panels
.. autofunction:: image_matchmaker.utils.plot_matching_qc
.. autofunction:: image_matchmaker.utils.plot_matching_qc_panels
.. autofunction:: image_matchmaker.utils.plot_landmark_overlay
.. autofunction:: image_matchmaker.mobie_export.export_to_mobie

Pipeline stages
---------------

The main entry function for each registration stage, in pipeline order:

.. autofunction:: image_matchmaker.prealignment.run_prealignment
.. autofunction:: image_matchmaker.rigid_alignment_elastix.run_rigid_alignment
.. autofunction:: image_matchmaker.cpd_nonrigid_registration.run_cpd
.. autofunction:: image_matchmaker.match_pointclouds.run_matching
.. autofunction:: image_matchmaker.elastix_deformable_pointset_registration.run_pointset_registration

Applying transforms
-------------------

.. autofunction:: image_matchmaker.apply_transform.apply_transform
