from sklearn.feature_selection import SelectKBest, f_classif 
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
# ----------------------------------------------------------------------
# FEATURE ENGINEERING  ✧  returns: X_fe (n_samples × n_features) + names
# ----------------------------------------------------------------------
from scipy.signal import savgol_filter, find_peaks
from simca.preprocessing import savgol_smooth
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, entropy

def build_feature_matrix(
        X_raw: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
        *,
        include_deriv=True,
        include_peak_stats=True,
        include_region_ratios=True,
        include_pca_features=True,
        include_entropy=True,
        include_percentiles=True,
        include_moments=True,
        n_pca_components=10
    ) -> Tuple[np.ndarray, List[str]]:
    """
    Turn raw spectra into an engineered feature matrix.

    Parameters
    ----------
    X_raw : (n_samples, n_channels) float
    wavenumbers : (n_channels,) float or None
    include_deriv, include_peak_stats, include_region_ratios : bool

    Returns
    -------
    X_fe      : (n_samples, n_features) float
    feat_names: list[str]
    """
    feat_blocks, names = [X_raw], ["orig"]

    # ── 1 · Savitzky–Golay derivatives ────────────────────────────────
    if include_deriv:
        d1 = np.array([savgol_filter(x, 15, 3, deriv=1) for x in X_raw])
        d2 = np.array([savgol_filter(x, 15, 3, deriv=2) for x in X_raw])
        feat_blocks += [d1, d2]
        names       += ["deriv1", "deriv2"]

    # ── 2 · Region-averaged ratios ────────────────────────────────────
    if include_region_ratios and wavenumbers is not None:
        thirds = np.array_split(np.arange(X_raw.shape[1]), 3)
        reg_means = [X_raw[:, idx].mean(1, keepdims=True) for idx in thirds]
        r12 = reg_means[0] / (reg_means[1]+1e-9)
        r13 = reg_means[0] / (reg_means[2]+1e-9)
        r23 = reg_means[1] / (reg_means[2]+1e-9)
        feat_blocks += [np.hstack([r12, r13, r23])]
        names       += ["ratios"]

    # ── 3 · Simple peak statistics ────────────────────────────────────
    if include_peak_stats:
        pk_feats = []
        for s in X_raw:
            pk, props = find_peaks(s, height=0.1, distance=8)
            h = props["peak_heights"] if len(pk) else np.array([0.0])
            pk_feats.append([
                len(pk),                     # #peaks
                h.max() if len(h) else 0.0,  # tallest peak
                h.mean() if len(h) else 0.0, # mean height
                s[pk].sum()                  # crude area
            ])
        feat_blocks += [np.asarray(pk_feats)]
        names       += ["peak_stats"]

    # 4 · Statistical moments
    if include_moments:
        moments = np.column_stack([
            X_raw.mean(axis=1),
            X_raw.std(axis=1),
            skew(X_raw, axis=1),
            kurtosis(X_raw, axis=1)
        ])
        feat_blocks.append(moments)
        names.append("moments")
    # 5 · PCA features
    if include_pca_features:
        pca = PCA(n_components=min(n_pca_components, X_raw.shape[1]))
        X_pca = pca.fit_transform(X_raw)
        feat_blocks.append(X_pca)
        names.append("pca")
    # 6 · Spectral entropy
    if include_entropy:
        X_norm = X_raw / (np.sum(X_raw, axis=1, keepdims=True) + 1e-12)
        ent = np.apply_along_axis(lambda x: entropy(x + 1e-12), 1, X_norm)
        feat_blocks.append(ent[:, None])
        names.append("entropy")
    # 7 · Intensity percentiles
    if include_percentiles:
        percentiles = np.percentile(X_raw, [10, 25, 50, 75, 90], axis=1).T
        feat_blocks.append(percentiles)
        names.append("percentiles")
    return np.hstack(feat_blocks), names


def engineer_and_select(
        X_raw: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        wavenumbers: Optional[np.ndarray] = None,
        k_best: int = 1200   # tune as you like
    ) -> Tuple[np.ndarray, SelectKBest]:
    """
    Returns engineered+reduced X plus the fitted selector (can be None).
    """
    X_fe, _ = build_feature_matrix(X_raw, wavenumbers)

    if y is None:           # test / predict time → use identity selector
        return X_fe, None

    k = min(k_best, X_fe.shape[1]-1)
    selector = SelectKBest(f_classif, k=k).fit(X_fe, y)
    return selector.transform(X_fe), selector
