import numpy as np


def zero_mean_unit_variance(img):
    return (img - np.mean(img)) / np.std(img)


def percentile_norm(img, pmin, pmax, eps=1e-10, channelwise=False):
    """
    Percentile normalization. Image is supposed to be C(Z)YX

    Args:
        img: _description_
        pmin: in percents
        pmax: in percents
        eps: _description_. Defaults to 1e-10.

    Returns:
        _description_
    """
    if channelwise:
        norm_img = np.zeros_like(img)
        for chan in range(norm_img.shape[0]):
            pmin_val = np.percentile(img[chan, ...], pmin)
            pmax_val = np.percentile(img[chan, ...], pmax)
            norm_img[chan, ...] = (img[chan, ...] - pmin_val) / (pmax_val - pmin_val + eps)
        return norm_img
    else:
        pmin = np.percentile(img, pmin)
        pmax = np.percentile(img, pmax)
        return (img - pmin) / (pmax - pmin + eps)
