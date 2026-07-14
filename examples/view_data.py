import napari
from image_matchmaker.utils import read_volume


seg_fixed = read_volume("./data/test/fixed_prealigned.n5", key="seg")
seg_moving = read_volume("./data/test/moving_prealigned.n5", key="seg")
# seg_moving = seg_moving[:, :, ::-1]

v = napari.Viewer()
v.add_labels(seg_fixed)
v.add_labels(seg_moving)
napari.run()