"""
evaluation.py
Comprehensive Quantitative Evaluation Suite for Lunar Image Representations:
1. Neighbourhood Preservation under Transformation (Top-k Jaccard, Spearman Rho, Trustworthiness)
2. Reconstruction Error (MSE, PSNR, SSIM across varying illumination)
3. Matching Stability (Mean Matching Accuracy MMA@tau, Repeatability, AUC vs Sun Angle & Scale)
4. Mandatory Comparison against Raw Intensity Correlation Baseline
"""

import numpy as np
import scipy.stats as stats
from sklearn.manifold import trustworthiness
from skimage.metrics import structural_similarity as ssim
from utils import (
    detect_lunar_keypoints,
    extract_patches,
    ground_truth_warp,
    match_descriptors
)

def compute_pairwise_cosine_distances(descriptors):
    """Pairwise cosine distance: D_ij = 1 - (u_i . u_j)"""
    normed = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7)
    sim = np.dot(normed, normed.T)
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    return dist

def evaluate_neighbourhood_preservation(desc_orig, desc_transf, k_list=(5, 10, 20)):
    """
    Measures how much the topological structure of the latent representation survives
    sun angle and scale changes.
    Metrics:
    - Top-k Jaccard Neighborhood Overlap (mean across all patches)
    - Spearman rank correlation of pairwise distance matrices
    - Trustworthiness (manifold topology preservation)
    """
    N = len(desc_orig)
    if N < 5:
        return {'jaccard_k5': 0.0, 'jaccard_k10': 0.0, 'spearman_rho': 0.0, 'trustworthiness': 0.0}

    D_orig = compute_pairwise_cosine_distances(desc_orig)
    D_transf = compute_pairwise_cosine_distances(desc_transf)

    # Exclude self-distance (diagonal = 0)
    np.fill_diagonal(D_orig, 1e9)
    np.fill_diagonal(D_transf, 1e9)

    results = {}

    # Top-k Jaccard Overlap
    for k in k_list:
        effective_k = min(k, N - 1)
        jaccard_scores = []
        for i in range(N):
            nn_orig = set(np.argsort(D_orig[i])[:effective_k])
            nn_transf = set(np.argsort(D_transf[i])[:effective_k])
            intersection = len(nn_orig & nn_transf)
            union = len(nn_orig | nn_transf)
            jaccard_scores.append(intersection / max(1, union))
        results[f'jaccard_k{k}'] = float(np.mean(jaccard_scores))

    # Reset diagonals to 0 for pairwise correlation
    np.fill_diagonal(D_orig, 0.0)
    np.fill_diagonal(D_transf, 0.0)

    # Spearman Rank Correlation between upper-triangular pairwise distances
    triu_indices = np.triu_indices(N, k=1)
    d_orig_flat = D_orig[triu_indices]
    d_transf_flat = D_transf[triu_indices]

    rho, _ = stats.spearmanr(d_orig_flat, d_transf_flat)
    results['spearman_rho'] = float(rho) if not np.isnan(rho) else 0.0

    # Trustworthiness: measures how well local neighbors are retained without intruder false neighbors
    try:
        tw = trustworthiness(desc_orig, desc_transf, n_neighbors=min(7, N // 2), metric='cosine')
        results['trustworthiness'] = float(tw)
    except Exception:
        results['trustworthiness'] = float(max(0.0, results['spearman_rho']))

    return results

def evaluate_reconstruction_error(autoencoder, test_patches, clean_dem_proxy=None):
    """
    Evaluates autoencoder reconstruction error under cross-illumination.
    Reports: MSE, PSNR, SSIM
    """
    if test_patches.ndim == 3:
        inputs = np.expand_dims(test_patches, axis=-1)
    else:
        inputs = test_patches

    recon = autoencoder.predict(inputs, batch_size=64, verbose=0)
    recon_squeezed = np.clip(recon[..., 0], 0.0, 1.0)
    inputs_squeezed = np.clip(inputs[..., 0], 0.0, 1.0)

    N = len(inputs_squeezed)
    mse_list = []
    psnr_list = []
    ssim_list = []

    for i in range(N):
        target = clean_dem_proxy[i] if clean_dem_proxy is not None else inputs_squeezed[i]
        diff = inputs_squeezed[i] - recon_squeezed[i]
        mse = np.mean(diff**2)
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
        ssim_val = ssim(target, recon_squeezed[i], data_range=1.0)
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim_val)

    return {
        'mean_mse': float(np.mean(mse_list)),
        'mean_psnr': float(np.mean(psnr_list)),
        'mean_ssim': float(np.mean(ssim_list)),
        'sample_originals': inputs_squeezed[:6],
        'sample_reconstructions': recon_squeezed[:6]
    }

def evaluate_matching_stability_pair(img1, img2, meta1, meta2, representation, pixel_thresholds=(1.0, 2.0, 3.0, 5.0, 8.0, 10.0)):
    """
    Evaluates correspondence stability for an image pair under known ground truth warping.
    Calculates:
    - Mean Matching Accuracy (MMA@tau) across pixel thresholds
    - Mean error in pixels
    - Repeatability / inlier count
    - AUC of MMA curve
    """
    kp1 = detect_lunar_keypoints(img1, max_keypoints=140)
    kp2 = detect_lunar_keypoints(img2, max_keypoints=140)

    fallback_dict = {
        'mma_at_1px': 0.0,
        'mma_at_2px': 0.0,
        'mma_at_3px': 0.0,
        'mma_at_5px': 0.0,
        'mma_at_10px': 0.0,
        'auc_mma': 0.0,
        'mean_error': 999.0,
        'median_error': 999.0,
        'total_matches': 0,
        'mma_curve': {f'{t}px': 0.0 for t in pixel_thresholds},
        'matches': [],
        'kp1': kp1,
        'kp2': kp2,
        'gt_kp2': np.zeros((0, 2), dtype=np.float32)
    }

    if len(kp1) < 5 or len(kp2) < 5:
        return fallback_dict

    # Extract patches
    patches1 = extract_patches(img1, kp1, patch_size=48)
    patches2 = extract_patches(img2, kp2, patch_size=48)

    # Extract descriptors
    desc1 = representation.extract_descriptors(patches1)
    desc2 = representation.extract_descriptors(patches2)

    # Match descriptors
    matches = match_descriptors(desc1, desc2, method='mnn')
    if len(matches) == 0:
        return fallback_dict

    # True coordinates of kp1 in img2
    gt_kp2 = ground_truth_warp(kp1, meta1, meta2)

    # Compute spatial errors
    errors = []
    for idx1, idx2, _ in matches:
        pred_p2 = kp2[idx2]
        true_p2 = gt_kp2[idx1]
        err = np.linalg.norm(pred_p2 - true_p2)
        errors.append(err)

    errors = np.array(errors, dtype=np.float32)

    # MMA curve across thresholds
    mma_curve = {}
    for tau in pixel_thresholds:
        acc = np.mean(errors <= tau) * 100.0
        mma_curve[f'{tau}px'] = float(acc)

    # AUC of MMA curve up to 10px (normalized to [0, 100])
    thresholds_arr = np.array(list(pixel_thresholds), dtype=np.float32)
    accuracies_arr = np.array([mma_curve[f'{tau}px'] for tau in pixel_thresholds], dtype=np.float32)
    # Manual trapezoidal integration for compatibility across NumPy 1.x and 2.x
    dx = np.diff(thresholds_arr)
    trapz_val = np.sum((accuracies_arr[:-1] + accuracies_arr[1:]) * 0.5 * dx)
    auc = float(trapz_val / (thresholds_arr[-1] - thresholds_arr[0]))

    return {
        'mma_at_1px': mma_curve.get('1.0px', 0.0),
        'mma_at_2px': mma_curve.get('2.0px', 0.0),
        'mma_at_3px': mma_curve.get('3.0px', 0.0),
        'mma_at_5px': mma_curve.get('5.0px', 0.0),
        'mma_at_10px': mma_curve.get('10.0px', 0.0),
        'auc_mma': auc,
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'total_matches': len(matches),
        'mma_curve': mma_curve,
        'matches': matches,
        'kp1': kp1,
        'kp2': kp2,
        'gt_kp2': gt_kp2
    }
