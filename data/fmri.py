"""
fMRI brain connectivity data loader.

Provides synthetic and real fMRI-like time series for causal discovery
benchmarks on brain functional connectivity.

Real data source: Human Connectome Project (HCP) resting-state fMRI
    - https://www.humanconnectome.org/
    - Requires nilearn for preprocessing

Synthetic data mimics known brain network structure:
    - Default Mode Network (DMN): mPFC, PCC, bilateral AG
    - Visual Network: V1, V2, MT
    - Motor Network: M1, SMA, PMC
    - Frontoparietal: dlPFC, IPL

Ground truth from tractography and lesion studies.
"""

import numpy as np
from typing import Tuple, Optional, List


# Known brain connectivity (simplified, from literature)
BRAIN_NETWORKS = {
    "DMN": ["mPFC", "PCC", "lAG", "rAG"],
    "Visual": ["V1", "V2", "MT"],
    "Motor": ["M1", "SMA", "PMC"],
    "Frontoparietal": ["dlPFC", "IPL"],
}

# Known directed connections (from effective connectivity studies)
KNOWN_CONNECTIONS = [
    # Visual hierarchy
    ("V1", "V2", 0.5),
    ("V2", "MT", 0.4),
    # DMN internal
    ("mPFC", "PCC", 0.35),
    ("PCC", "lAG", 0.3),
    ("PCC", "rAG", 0.3),
    # Motor
    ("SMA", "M1", 0.45),
    ("PMC", "SMA", 0.3),
    # Cross-network
    ("dlPFC", "mPFC", 0.2),
    ("MT", "IPL", 0.25),
]


def generate_synthetic_fmri(
    n_rois: int = 12,
    T: int = 1200,
    tr: float = 0.72,
    noise_std: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic fMRI-like data with known brain connectivity.

    Mimics properties of real fMRI:
    - Hemodynamic response function (HRF) blurring
    - High noise (SNR ~1-3)
    - Autocorrelation from HRF
    - Known directed connectivity between ROIs

    Args:
        n_rois: Number of brain regions (max 12).
        T: Number of time points (TRs).
        tr: Repetition time in seconds.
        noise_std: Noise level (fMRI is noisy).
        seed: Random seed.

    Returns:
        X: Time series (T, n_rois).
        G_true: Ground truth adjacency matrix.
        roi_names: List of ROI names.
    """
    rng = np.random.RandomState(seed)

    all_rois = []
    for network_rois in BRAIN_NETWORKS.values():
        all_rois.extend(network_rois)
    roi_names = all_rois[:n_rois]
    K = len(roi_names)

    name_to_idx = {name: i for i, name in enumerate(roi_names)}

    # Build VAR coefficient matrix
    A = np.diag([0.6] * K)  # Strong autocorrelation (HRF persistence)

    G_true = np.zeros((K, K))
    for src, tgt, strength in KNOWN_CONNECTIONS:
        if src in name_to_idx and tgt in name_to_idx:
            i, j = name_to_idx[tgt], name_to_idx[src]
            A[i, j] = strength * 0.5  # Scale down for stability
            G_true[i, j] = 1.0

    # Ensure stability
    eigvals = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 0.95:
        A = A * (0.94 / max_eig)

    # Generate neural signal (VAR process)
    neural = np.zeros((T * 2, K))  # Oversample for HRF convolution
    for t in range(1, T * 2):
        neural[t] = A @ neural[t - 1] + rng.randn(K) * 0.2

    # Convolve with HRF (simplified canonical HRF)
    hrf = _canonical_hrf(tr, duration=30.0)
    fmri_signal = np.zeros((T * 2, K))
    for k in range(K):
        fmri_signal[:, k] = np.convolve(neural[:, k], hrf)[:T * 2]

    # Downsample to TR resolution and add measurement noise
    X = fmri_signal[:T] + rng.randn(T, K) * noise_std

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, G_true, roi_names


def _canonical_hrf(tr: float, duration: float = 30.0) -> np.ndarray:
    """
    Generate a canonical hemodynamic response function.

    Uses the difference of two gamma functions (SPM-style).
    """
    dt = tr
    t = np.arange(0, duration, dt)

    # Parameters for canonical HRF
    a1, b1 = 6.0, 1.0   # Peak at ~6s
    a2, b2 = 16.0, 1.0  # Undershoot at ~16s
    c = 1.0 / 6.0        # Ratio of undershoot

    from scipy.stats import gamma as gamma_dist
    h1 = gamma_dist.pdf(t, a1, scale=b1)
    h2 = gamma_dist.pdf(t, a2, scale=b2)
    hrf = h1 - c * h2

    # Normalize
    hrf = hrf / np.max(hrf)
    return hrf


def load_hcp_fmri(
    data_dir: str = "data/fmri_raw",
    n_rois: int = 15,
    atlas: str = "schaefer",
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Load real HCP resting-state fMRI data.

    Requires nilearn and downloaded HCP data.

    Args:
        data_dir: Directory with preprocessed fMRI data.
        n_rois: Number of ROIs to extract.
        atlas: Parcellation atlas ("schaefer" or "aal").

    Returns:
        X: Time series (T, n_rois) or None if data not found.
        roi_names: List of ROI names.
    """
    try:
        import os
        import glob

        # Look for preprocessed .npy files first
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        if npy_files:
            X = np.load(npy_files[0])
            if X.ndim == 2 and X.shape[1] >= n_rois:
                X = X[:, :n_rois]
                roi_names = [f"ROI_{i+1}" for i in range(n_rois)]
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                return X, roi_names

        # Try nilearn if installed
        from nilearn import datasets
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=100)
        print(f"  Atlas loaded: {atlas_data['description'][:50]}...")
        # Would need actual fMRI data files to proceed
        return None, []

    except ImportError:
        print("  nilearn not installed. Using synthetic fMRI data.")
        return None, []
    except Exception as e:
        print(f"  Could not load fMRI data: {e}")
        return None, []
