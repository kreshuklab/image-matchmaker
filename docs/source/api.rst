API Reference
=============

This page documents the most useful functions for calling image-matchmaker from
Python. For usage patterns and examples, see the Python API section of the
:doc:`Usage <usage>` page.

The helpers below are re-exported from ``image_matchmaker.utils``; the pipeline-stage
functions live in their respective modules.

I/O
---

.. autofunction:: matchmaker.utils.read_volume
.. autofunction:: matchmaker.utils.write_volume
.. autofunction:: matchmaker.utils.get_attrs
.. autofunction:: matchmaker.utils.set_attrs
.. autofunction:: matchmaker.utils.load_config
.. autofunction:: matchmaker.utils.download_file

Transforms
----------

.. autofunction:: matchmaker.utils.rotate_img
.. autofunction:: matchmaker.utils.resample_volume
.. autofunction:: matchmaker.utils.get_transformation_matrix
.. autofunction:: matchmaker.utils.write_transform_dict
.. autofunction:: matchmaker.utils.read_transform_dict

Point clouds
------------

.. autofunction:: matchmaker.utils.extract_centroids
.. autofunction:: matchmaker.utils.create_pcd
.. autofunction:: matchmaker.utils.sparse_ilp_matching
.. autofunction:: matchmaker.utils.hungarian_matching
.. autofunction:: matchmaker.utils.sinkhorn_matching

Visualization
-------------

.. autofunction:: matchmaker.utils.plot_three_slices
.. autofunction:: matchmaker.utils.plot_overlay
.. autofunction:: matchmaker.utils.overlay_pcds
.. autofunction:: matchmaker.utils.visualize_displacement_field
.. autofunction:: matchmaker.utils.plot_matching_qc
.. autofunction:: matchmaker.mobie_export.export_to_mobie

Pipeline stages
---------------

The main entry function for each registration stage, in pipeline order:

.. autofunction:: matchmaker.prealignment.run_prealignment
.. autofunction:: matchmaker.rigid_alignment_elastix.run_rigid_alignment
.. autofunction:: matchmaker.cpd_nonrigid_registration.run_cpd
.. autofunction:: matchmaker.match_pointclouds.run_matching
.. autofunction:: matchmaker.elastix_deformable_pointset_registration.run_pointset_registration

Applying transforms
-------------------

.. autofunction:: matchmaker.apply_transform.apply_transform
