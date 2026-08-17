import numpy as np
import open3d as o3d

from image_matchmaker.utils import (
    create_pcd,
    extract_centroids,
    itk_scalar_img,
    pcd_to_elastix,
)

RESOLUTION_ZYX = [0.7, 0.5, 0.3]
VOXEL_ZYX = (5, 3, 1)


def test_image_and_pointset_agree_on_physical_space(tmp_path):
    seg = np.zeros((7, 5, 3), np.uint16)
    seg[VOXEL_ZYX] = 1  # single voxel, so its centroid is exactly its index

    # point-set path
    labels, coords = extract_centroids(seg, RESOLUTION_ZYX)
    pcd_path = tmp_path / "pcd.pcd"
    o3d.t.io.write_point_cloud(str(pcd_path), create_pcd(coords, labels), write_ascii=True)
    points_path = tmp_path / "points.txt"
    pcd_to_elastix(str(pcd_path), str(points_path))
    from_pointset = [float(v) for v in points_path.read_text().split("\n")[2].split()]

    # image path; ITK indexes (x, y, z), so the numpy index is reversed
    img = itk_scalar_img(seg.astype(np.float32), RESOLUTION_ZYX)
    from_image = list(img.TransformIndexToPhysicalPoint(VOXEL_ZYX[::-1]))

    np.testing.assert_allclose(from_pointset, from_image)

    # check the value too: two conventions flipped together would pass the assert above
    z, y, x = VOXEL_ZYX
    rz, ry, rx = RESOLUTION_ZYX
    np.testing.assert_allclose(from_image, [x * rx, y * ry, z * rz])
