import numpy as np
from scipy.ndimage import laplace


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    return img.astype(np.float32)


def laplacian_variance(img_gray: np.ndarray) -> float:
    lap = laplace(img_gray.astype(np.float32))
    return float(np.var(lap))


def fft_high_freq_energy(img_gray: np.ndarray, cutoff_fraction: float = 0.3) -> float:
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((Y - cy) / cy) ** 2 + ((X - cx) / cx) ** 2)
    high_freq_mask = dist > cutoff_fraction
    total_energy = magnitude.sum() + 1e-10
    return float(magnitude[high_freq_mask].sum() / total_energy)
