import z5py
import napari

with z5py.File("/Users/marei/git-repositories/matchmaker/examples/data/test/mobie_project/platy1_muscles_stardist/images/ome-zarr/original.ome.zarr", "r") as f:
    seg = f["s0"][:]

v = napari.Viewer()
v.add_image(seg)
napari.run()