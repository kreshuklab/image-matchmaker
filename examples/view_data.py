from matchmaker.n5_utils import read_volume
import napari

seg_fixed = read_volume("./data/test/platy1_muscles_stardist_fixed_prealigned.n5", key="seg")
seg_moving = read_volume("./data/test/platy1_muscles_stardist_moving_prealigned.n5", key="seg")
# seg_moving = seg_moving[:, :, ::-1]

v = napari.Viewer()
v.add_labels(seg_fixed)
v.add_labels(seg_moving)
napari.run()