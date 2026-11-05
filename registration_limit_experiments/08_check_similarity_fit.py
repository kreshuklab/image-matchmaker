"""Check NucleiSky vs IMM results: overlay for ji-0.25 and similarity-fit analysis for all settings.

For each warped image, a similarity transform (Umeyama) is fitted from warped to fixed landmark
centroids. Its scale/rotation show what is left of a global misalignment, its residual what is left
after removing it. The "oracle" fits the best similarity from the unwarped full moving image to
the fixed image: the lowest LRE any similarity-only method can reach on this elastic data.
Run from the output folder ($OUT) with imm_env, after steps 05 and 07:
    python 08_check_similarity_fit.py <overlay.png> > results/similarity_fit.tsv
Also writes results/ji-0.25/overlay_fixed_warped.pdf.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import center_of_mass

from image_matchmaker.utils import read_volume

OUT_PNG = Path(sys.argv[1])
FIXED = tifffile.imread("data/platy1_muscles_stardist_fixed_rotated.tif").astype(np.int32)
LANDMARKS = np.loadtxt("metric_ids.txt", dtype=int)
MOVING = "platy1_muscles_stardist_elastic"


def centroids(seg, ids):
    return np.array(center_of_mass(seg > 0, seg, ids))


def umeyama(src, dst):
    """Similarity dst ~ s R src + t. Returns s, R, t, residual distances."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ sign @ vt
    s = (d * np.diag(sign)).sum() / a.var(0).sum()
    t = mu_d - s * r @ mu_s
    res = np.linalg.norm(dst - (s * (r @ src.T).T + t), axis=1)
    return s, r, t, res


def angle(r):
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


fixed_c = {}
print("setting\tmethod\tLRE\tfit_scale\tfit_rot_deg\tfit_shift\tLRE_after_fit\toracle_similarity_LRE")
for d in sorted(Path("results").iterdir()):
    tag = d.name
    if not (d / "lre_imm_full.txt").exists():
        continue
    full = tifffile.imread(f"data/deformed_data_{tag}/{MOVING}_full.tif").astype(np.int32)
    ids_o = np.intersect1d(LANDMARKS, np.unique(full))
    _, _, _, res_o = umeyama(centroids(full, ids_o), centroids(FIXED, ids_o))
    warped = {
        "nucleisky": tifffile.imread(d / "nucleisky_full.tif").astype(np.int32),
        "imm": read_volume(str(d / "moving_image.n5"), "full_pointset_alignment_transform").astype(np.int32),
    }
    for method, w in warped.items():
        ids = np.loadtxt(d / f"lre_{method}_full.txt", dtype=float)[:, 0].astype(int)
        cw, cf = centroids(w, ids), centroids(FIXED, ids)
        s, r, t, res = umeyama(cw, cf)
        lre = np.linalg.norm(cw - cf, axis=1).mean()
        shift = np.linalg.norm(cf.mean(0) - cw.mean(0))
        print(f"{tag}\t{method}\t{lre:.2f}\t{s:.3f}\t{angle(r):.2f}\t{shift:.2f}\t{res.mean():.2f}\t{res_o.mean():.2f}",
              flush=True)

    if tag == "ji-0.25":
        center = [int(c) for c in center_of_mass(FIXED > 0)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for row, (method, w) in enumerate(warped.items()):
            for col, ax_name in enumerate("zyx"):
                sl = [slice(None)] * 3
                sl[col] = center[col]
                f, m = FIXED[tuple(sl)] > 0, w[tuple(sl)] > 0
                rgb = np.stack([f, m, f], -1).astype(float)  # fixed magenta, warped green, overlap white
                axes[row, col].imshow(rgb)
                axes[row, col].set_title(f"{method}, {ax_name} = {center[col]}")
                axes[row, col].axis("off")
        fig.suptitle("ji-0.25: fixed (magenta) vs warped full moving (green), overlap white")
        fig.tight_layout()
        fig.savefig(d / "overlay_fixed_warped.pdf")
        fig.savefig(OUT_PNG, dpi=80)
