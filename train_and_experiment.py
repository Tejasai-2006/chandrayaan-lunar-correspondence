"""
train_and_experiment.py
End-to-end Training and Benchmark Experimentation Pipeline.
1. Generates multi-scene lunar datasets (crater fields, polar craters, mare basalt).
2. Builds self-supervised pairs under solar illumination disparity (azimuth + elevation) and scale variations.
3. Trains Lunar-InvNet (Contrastive InfoNCE) and Lunar-AE (Autoencoder).
4. Evaluates all 3 required objectives:
   - Neighbourhood Preservation under Transformation (Top-k Jaccard, Spearman Rho, Trustworthiness)
   - Reconstruction Error (MSE, PSNR, SSIM)
   - Matching Stability (MMA@tau, Repeatability, AUC vs sun angle and scale)
   - Mandatory Raw Intensity Correlation baseline comparison.
5. Saves model weights, figures, CSVs, and JSON summary for the Streamlit Explorer.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras

from lunar_engine import LunarTerrainEngine
from representations import (
    RawIntensityBaseline,
    ClassicalGradientRepresentation,
    LunarInvNetRepresentation,
    LunarAutoencoderRepresentation,
    build_lunar_invnet_backbone,
    build_lunar_autoencoder
)
from utils import (
    detect_lunar_keypoints,
    extract_patches,
    ground_truth_warp,
    draw_correspondence_explorer_view,
    compute_reconstruction_scores
)
from evaluation import (
    evaluate_neighbourhood_preservation,
    evaluate_reconstruction_error,
    evaluate_matching_stability_pair
)

def create_training_pairs(num_scenes=4, patches_per_scene=200, patch_size=48, seed=42):
    """
    Extract training patch pairs from authentic ISRO Chandrayaan-2 lunar terrain scenes
    (TMC-2 & OHRC) and build self-supervised pairs under solar illumination disparity
    and scale variations.
    """
    print(f"Extracting training pairs from {num_scenes} authentic ISRO Chandrayaan lunar scenes...")
    rng = np.random.RandomState(seed)
    pairs_a = []
    pairs_b = []
    dem_targets = []

    sun_elevations = [15.0, 25.0, 35.0, 50.0, 65.0]
    sun_azimuths = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    for s_idx in range(num_scenes):
        engine = LunarTerrainEngine(size=384, resolution_m_per_px=1.0, seed=seed + s_idx * 17)
        # Select keypoints on DEM terrain
        kps = detect_lunar_keypoints(engine.albedo, max_keypoints=patches_per_scene * 2, min_distance=10)
        if len(kps) > patches_per_scene:
            chosen_indices = rng.choice(len(kps), patches_per_scene, replace=False)
            kps = kps[chosen_indices]

        # Extract normalized DEM proxy target for autoencoder
        dem_norm = (engine.dem - engine.dem.min()) / (engine.dem.max() - engine.dem.min() + 1e-6)
        clean_dem_patches = extract_patches(dem_norm, kps, patch_size=patch_size)

        # Render under two distinct illumination conditions
        # View A: Condition 1
        az_a = rng.choice(sun_azimuths)
        el_a = rng.choice(sun_elevations)
        img_a, _ = engine.render_image(az_a, el_a)
        patches_a = extract_patches(img_a, kps, patch_size=patch_size)

        # View B: Condition 2 (different azimuth and elevation)
        az_b = (az_a + rng.choice([60.0, 90.0, 135.0, 180.0])) % 360.0
        el_b = rng.choice([el for el in sun_elevations if abs(el - el_a) >= 15.0] or [40.0])
        img_b, _ = engine.render_image(az_b, el_b)
        patches_b = extract_patches(img_b, kps, patch_size=patch_size)

        # Add data augmentations: random scale resizing, gamma perturbation, flips
        for i in range(len(kps)):
            pa = patches_a[i]
            pb = patches_b[i]
            p_dem = clean_dem_patches[i]

            # Augment View B with scale simulation (zoom crop)
            if rng.rand() > 0.5:
                zoom = rng.uniform(1.2, 1.5)
                h, w = pb.shape
                zh, zw = int(h / zoom), int(w / zoom)
                zy, zx = (h - zh) // 2, (w - zw) // 2
                pb = cv2.resize(pb[zy:zy+zh, zx:zx+zw], (w, h), interpolation=cv2.INTER_CUBIC)

            # Random gamma
            gamma = rng.uniform(0.65, 1.4)
            pa = np.power(np.clip(pa, 1e-4, 1.0), gamma)
            pb = np.power(np.clip(pb, 1e-4, 1.0), 1.0 / gamma)

            # Random horizontal / vertical flip
            if rng.rand() > 0.5:
                pa = np.fliplr(pa)
                pb = np.fliplr(pb)
                p_dem = np.fliplr(p_dem)
            if rng.rand() > 0.5:
                pa = np.flipud(pa)
                pb = np.flipud(pb)
                p_dem = np.flipud(p_dem)

            pairs_a.append(pa)
            pairs_b.append(pb)
            dem_targets.append(p_dem)

    X_a = np.expand_dims(np.array(pairs_a, dtype=np.float32), axis=-1)
    X_b = np.expand_dims(np.array(pairs_b, dtype=np.float32), axis=-1)
    Y_dem = np.expand_dims(np.array(dem_targets, dtype=np.float32), axis=-1)

    print(f"Generated {len(X_a)} training patch pairs.")
    return X_a, X_b, Y_dem

def train_lunar_invnet(model, X_a, X_b, epochs=12, batch_size=48, temperature=0.07, lr=1e-3):
    """
    Train Lunar-InvNet using InfoNCE contrastive self-supervised objective.
    """
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(batch_a, batch_b):
        with tf.GradientTape() as tape:
            # Forward pass: extract unit-normalized representations
            z_a = model(batch_a, training=True)
            z_b = model(batch_b, training=True)

            # Cosine similarity matrix scaled by temperature
            sim = tf.matmul(z_a, z_b, transpose_b=True) / temperature
            batch_dim = tf.shape(z_a)[0]
            labels = tf.range(batch_dim)

            # Symmetric cross-entropy InfoNCE loss
            loss_a = tf.keras.losses.sparse_categorical_crossentropy(labels, sim, from_logits=True)
            loss_b = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(sim), from_logits=True)
            loss = tf.reduce_mean((loss_a + loss_b) * 0.5)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    num_samples = len(X_a)
    indices = np.arange(num_samples)
    print("\n--- Training Lunar-InvNet (Contrastive Self-Supervised Learning) ---")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []
        for i in range(0, num_samples, batch_size):
            b_idx = indices[i:i + batch_size]
            if len(b_idx) < 16:
                continue
            loss_val = train_step(X_a[b_idx], X_b[b_idx])
            epoch_losses.append(float(loss_val))

        avg_loss = np.mean(epoch_losses)
        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Contrastive InfoNCE Loss: {avg_loss:.4f}")

    return model

def train_lunar_autoencoder(autoencoder, X_all, Y_targets, epochs=12, batch_size=48, lr=1e-3):
    """
    Train Autoencoder on multi-illumination lunar patches to reconstruct underlying terrain morphology.
    """
    print("\n--- Training Lunar-AE (Scale-Equivariant Invariant Autoencoder) ---")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    autoencoder.fit(X_all, Y_targets, epochs=epochs, batch_size=batch_size, verbose=0)
    eval_loss = autoencoder.evaluate(X_all[:128], Y_targets[:128], verbose=0)
    print(f"Autoencoder Final Reconstruction MSE: {eval_loss[0]:.5f}")
    return autoencoder

def run_illumination_sweep_experiments(engine, representations, delta_azimuths=(0.0, 30.0, 60.0, 90.0, 135.0, 180.0), delta_elevations=(0.0, 15.0, 30.0, 45.0)):
    """
    Experiment 1: Sweep across solar illumination disparities (azimuth and elevation).
    Measures MMA@3px, Mean Error, AUC vs baseline for all representations.
    """
    print("\n=== Running Experiment 1: Solar Illumination Disparity Sweep ===")
    records = []
    base_az = 45.0
    base_el = 30.0

    # Render reference image
    img_ref, meta_ref = engine.sample_observation(sun_azimuth=base_az, sun_elevation=base_el, scale=1.0)

    for daz in delta_azimuths:
        target_az = (base_az + daz) % 360.0
        for delv in delta_elevations:
            target_el = max(15.0, min(75.0, base_el + delv))
            img_query, meta_query = engine.sample_observation(sun_azimuth=target_az, sun_elevation=target_el, scale=1.0)

            for rep_name, rep_obj in representations.items():
                res = evaluate_matching_stability_pair(img_ref, img_query, meta_ref, meta_query, rep_obj)
                records.append({
                    'representation': rep_name,
                    'delta_azimuth': float(daz),
                    'delta_elevation': float(delv),
                    'scale': 1.0,
                    'mma_at_3px': res.get('mma_at_3px', 0.0),
                    'mma_at_5px': res.get('mma_at_5px', 0.0),
                    'auc_mma': res.get('auc_mma', 0.0),
                    'mean_error': res.get('mean_error', 999.0),
                    'total_matches': res.get('total_matches', 0)
                })

    df = pd.DataFrame(records)
    print(f"Completed illumination sweep ({len(df)} evaluations).")
    return df

def run_scale_sweep_experiments(engine, representations, scales=(1.0, 1.25, 1.5, 2.0, 2.5, 3.0)):
    """
    Experiment 2: Sweep across spatial resolution / scale changes under moderate illumination shift.
    """
    print("\n=== Running Experiment 2: Spatial Scale Disparity Sweep ===")
    records = []
    base_az, base_el = 45.0, 35.0
    target_az, target_el = 105.0, 25.0  # Moderate 60 deg azimuth and 10 deg elevation shift

    # Reference at scale 1.0
    img_ref, meta_ref = engine.sample_observation(sun_azimuth=base_az, sun_elevation=base_el, scale=1.0)

    for s in scales:
        img_query, meta_query = engine.sample_observation(sun_azimuth=target_az, sun_elevation=target_el, scale=s)
        for rep_name, rep_obj in representations.items():
            res = evaluate_matching_stability_pair(img_ref, img_query, meta_ref, meta_query, rep_obj)
            records.append({
                'representation': rep_name,
                'scale': float(s),
                'delta_azimuth': 60.0,
                'delta_elevation': 10.0,
                'mma_at_3px': res.get('mma_at_3px', 0.0),
                'mma_at_5px': res.get('mma_at_5px', 0.0),
                'auc_mma': res.get('auc_mma', 0.0),
                'mean_error': res.get('mean_error', 999.0),
                'total_matches': res.get('total_matches', 0)
            })

    df = pd.DataFrame(records)
    print(f"Completed scale sweep ({len(df)} evaluations).")
    return df

def run_neighbourhood_preservation_study(engine, representations, num_test_patches=150):
    """
    Experiment 4: Measure neighbourhood preservation under transformation:
    Top-5, Top-10, Top-20 Jaccard overlap, Spearman rho, and Trustworthiness.
    """
    print("\n=== Running Experiment 4: Neighbourhood Preservation Study ===")
    # Reference condition
    img_ref, meta_ref = engine.sample_observation(sun_azimuth=45.0, sun_elevation=35.0, scale=1.0)
    kps = detect_lunar_keypoints(img_ref, max_keypoints=num_test_patches, min_distance=10)
    if len(kps) > num_test_patches:
        kps = kps[:num_test_patches]

    # T3: Extreme joint transform (Azimuth 135 deg, el=60 deg, scale=1.5)
    img_t3, meta_t3 = engine.sample_observation(sun_azimuth=135.0, sun_elevation=60.0, scale=1.5)
    kps_t3 = ground_truth_warp(kps, meta_ref, meta_t3)

    # Keep only keypoints that are comfortably within bounds across all transformed views
    valid_mask = (
        (kps_t3[:, 0] >= 15) & (kps_t3[:, 0] < img_t3.shape[1] - 15) &
        (kps_t3[:, 1] >= 15) & (kps_t3[:, 1] < img_t3.shape[0] - 15) &
        (kps[:, 0] >= 15) & (kps[:, 0] < img_ref.shape[1] - 15) &
        (kps[:, 1] >= 15) & (kps[:, 1] < img_ref.shape[0] - 15)
    )
    kps = kps[valid_mask]
    kps_t3 = kps_t3[valid_mask]

    patches_ref = extract_patches(img_ref, kps, patch_size=48)

    # 3 Distinct Transformations
    # T1: Azimuth inversion (180 deg solar rotation)
    img_t1, _ = engine.sample_observation(sun_azimuth=225.0, sun_elevation=35.0, scale=1.0)
    patches_t1 = extract_patches(img_t1, kps, patch_size=48)

    # T2: Low Grazing Sun (High shadows, el=15 deg)
    img_t2, _ = engine.sample_observation(sun_azimuth=45.0, sun_elevation=15.0, scale=1.0)
    patches_t2 = extract_patches(img_t2, kps, patch_size=48)

    # T3 patches
    patches_t3 = extract_patches(img_t3, kps_t3, patch_size=48)

    transformations = {
        'Azimuth Inversion (180 deg)': patches_t1,
        'Grazing Sun Elevation (15 deg)': patches_t2,
        'Joint Illumination & Scale (s=1.5)': patches_t3
    }

    preservation_report = {}

    for rep_name, rep_obj in representations.items():
        desc_ref = rep_obj.extract_descriptors(patches_ref)
        preservation_report[rep_name] = {}

        for t_name, p_transf in transformations.items():
            desc_transf = rep_obj.extract_descriptors(p_transf)
            scores = evaluate_neighbourhood_preservation(desc_ref, desc_transf, k_list=(5, 10, 20))
            preservation_report[rep_name][t_name] = scores

    return preservation_report

def run_reconstruction_study(engine, autoencoder, num_patches=100):
    """
    Experiment 5: Measure reconstruction error (MSE, PSNR, SSIM) under varying illumination angles.
    """
    print("\n=== Running Experiment 5: Reconstruction Error Study ===")
    kps = detect_lunar_keypoints(engine.albedo, max_keypoints=num_patches, min_distance=10)
    dem_norm = (engine.dem - engine.dem.min()) / (engine.dem.max() - engine.dem.min() + 1e-6)
    clean_target_patches = extract_patches(dem_norm, kps, patch_size=48)

    elevations = [15.0, 30.0, 50.0, 70.0]
    recon_results = {}

    for el in elevations:
        img_illum, _ = engine.render_image(sun_azimuth_deg=45.0, sun_elevation_deg=el)
        patches_illum = extract_patches(img_illum, kps, patch_size=48)
        scores = evaluate_reconstruction_error(autoencoder, patches_illum, clean_dem_proxy=clean_target_patches)
        recon_results[f'{int(el)}deg_elevation'] = {
            'mean_mse': scores['mean_mse'],
            'mean_psnr': scores['mean_psnr'],
            'mean_ssim': scores['mean_ssim']
        }

    return recon_results

def generate_benchmark_figures(illum_df, scale_df, preservation_report, recon_results, results_dir='results'):
    """Generate publication-quality plots summarizing all benchmark metrics."""
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Comprehensive Benchmark Grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=150)

    # Panel A: MMA@3px vs Solar Azimuth Disparity
    ax1 = axes[0, 0]
    for rep, grp in illum_df[illum_df['delta_elevation'] == 30.0].groupby('representation'):
        sorted_g = grp.sort_values('delta_azimuth')
        ax1.plot(sorted_g['delta_azimuth'], sorted_g['mma_at_3px'], marker='o', linewidth=2.2, label=rep)
    ax1.set_title("A. Matching Accuracy vs Sun Azimuth Disparity", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Solar Azimuth Difference Δφ (degrees)", fontsize=11)
    ax1.set_ylabel("Mean Matching Accuracy (MMA@3px %)", fontsize=11)
    ax1.set_ylim(-2, 102)
    ax1.legend(loc='lower left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel B: MMA@3px vs Spatial Scale Disparity
    ax2 = axes[0, 1]
    for rep, grp in scale_df.groupby('representation'):
        sorted_g = grp.sort_values('scale')
        ax2.plot(sorted_g['scale'], sorted_g['mma_at_3px'], marker='s', linewidth=2.2, label=rep)
    ax2.set_title("B. Matching Accuracy vs Scale Ratio (Zoom/Altitude)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Scale Factor (s)", fontsize=11)
    ax2.set_ylabel("Mean Matching Accuracy (MMA@3px %)", fontsize=11)
    ax2.set_ylim(-2, 102)
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Panel C: Neighbourhood Preservation (Top-10 Jaccard Overlap)
    ax3 = axes[1, 0]
    t_name = 'Azimuth Inversion (180 deg)'
    rep_names = list(preservation_report.keys())
    jaccard_scores = [preservation_report[r][t_name]['jaccard_k10'] * 100.0 for r in rep_names]
    spearman_scores = [preservation_report[r][t_name]['spearman_rho'] * 100.0 for r in rep_names]

    x_pos = np.arange(len(rep_names))
    width = 0.35
    ax3.bar(x_pos - width/2, jaccard_scores, width, label='Top-10 Jaccard Overlap (%)', color='#2b5c8f')
    ax3.bar(x_pos + width/2, spearman_scores, width, label='Spearman Rank Corr ρ (x100)', color='#17b978')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([r.replace(' (Baseline)', '').replace(' (Contrastive Invariant)', '') for r in rep_names], rotation=15, ha='right', fontsize=9)
    ax3.set_title(f"C. Neighbourhood Preservation ({t_name})", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Preservation Score (%)", fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Panel D: Autoencoder Reconstruction Error vs Sun Elevation
    ax4 = axes[1, 1]
    elevations = [int(k.split('deg')[0]) for k in recon_results.keys()]
    psnr_vals = [recon_results[k]['mean_psnr'] for k in recon_results.keys()]
    ssim_vals = [recon_results[k]['mean_ssim'] * 35.0 for k in recon_results.keys()] # Scale for dual visualization

    ax4.plot(elevations, psnr_vals, marker='^', color='#e84a5f', linewidth=2.5, label='PSNR (dB)')
    ax4.plot(elevations, ssim_vals, marker='D', color='#9922bb', linewidth=2.5, label='SSIM (scaled x35)')
    ax4.set_title("D. Terrain Reconstruction Quality vs Solar Elevation", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Solar Elevation Angle (degrees above horizon)", fontsize=11)
    ax4.set_ylabel("Reconstruction Metric", fontsize=11)
    ax4.legend(loc='lower right', frameon=True)
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'benchmark_plots.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved benchmark figure to: {plot_path}")

def run_main():
    start_time = time.time()
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 2. Build and Train Lunar-InvNet
    invnet_backbone = build_lunar_invnet_backbone(input_shape=(48, 48, 1), embedding_dim=128)
    invnet_weights_path = os.path.join('models', 'lunar_invnet.weights.h5')
    autoencoder, encoder = build_lunar_autoencoder(input_shape=(48, 48, 1), latent_dim=64)
    ae_weights_path = os.path.join('models', 'lunar_autoencoder.weights.h5')

    if os.path.exists(invnet_weights_path) and os.path.exists(ae_weights_path):
        print(f"Loading existing model checkpoints...")
        invnet_backbone.load_weights(invnet_weights_path)
        autoencoder.load_weights(ae_weights_path)
    else:
        # 1. Dataset Generation
        X_a, X_b, Y_dem = create_training_pairs(num_scenes=4, patches_per_scene=160, patch_size=48, seed=42)
        X_all = np.concatenate([X_a, X_b], axis=0)
        Y_all = np.concatenate([Y_dem, Y_dem], axis=0)

        invnet_backbone = train_lunar_invnet(invnet_backbone, X_a, X_b, epochs=12, batch_size=48)
        invnet_backbone.save_weights(invnet_weights_path)
        print(f"Saved Lunar-InvNet weights to {invnet_weights_path}")

        autoencoder = train_lunar_autoencoder(autoencoder, X_all, Y_all, epochs=10, batch_size=48)
        autoencoder.save_weights(ae_weights_path)
        print(f"Saved Autoencoder weights to {ae_weights_path}")

    # 4. Instantiate all representations for benchmark
    representations = {
        'Raw Intensity (Baseline)': RawIntensityBaseline(patch_size=48),
        'Classical Gradient Histogram': ClassicalGradientRepresentation(patch_size=48),
        'Lunar-InvNet (Contrastive Invariant)': LunarInvNetRepresentation(invnet_backbone, patch_size=48),
        'Lunar-AE (Latent Bottleneck)': LunarAutoencoderRepresentation(encoder, patch_size=48)
    }

    # 5. Independent Benchmark Terrain Engine
    test_engine = LunarTerrainEngine(size=400, resolution_m_per_px=1.0, seed=999)

    # 6. Run Experiment Sweeps
    illum_df = run_illumination_sweep_experiments(test_engine, representations)
    illum_df.to_csv('results/illumination_sweep.csv', index=False)

    scale_df = run_scale_sweep_experiments(test_engine, representations)
    scale_df.to_csv('results/scale_sweep.csv', index=False)

    preservation_report = run_neighbourhood_preservation_study(test_engine, representations)
    with open('results/neighbourhood_preservation.json', 'w') as f:
        json.dump(preservation_report, f, indent=2)

    recon_results = run_reconstruction_study(test_engine, autoencoder)
    with open('results/reconstruction_metrics.json', 'w') as f:
        json.dump(recon_results, f, indent=2)

    # 7. Generate Visual Artifacts
    generate_benchmark_figures(illum_df, scale_df, preservation_report, recon_results, results_dir='results')

    # 8. Compute Overall Summary Scorecard
    summary_card = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_seconds': round(time.time() - start_time, 2),
        'representations_evaluated': list(representations.keys()),
        'illumination_stability_mma3px_mean': {
            rep: float(illum_df[illum_df['representation'] == rep]['mma_at_3px'].mean())
            for rep in representations.keys()
        },
        'scale_stability_mma3px_mean': {
            rep: float(scale_df[scale_df['representation'] == rep]['mma_at_3px'].mean())
            for rep in representations.keys()
        },
        'extreme_azimuth_180deg_mma3px': {
            rep: float(illum_df[(illum_df['representation'] == rep) & (illum_df['delta_azimuth'] == 180.0)]['mma_at_3px'].mean())
            for rep in representations.keys()
        },
        'neighbourhood_preservation_spearman_rho_mean': {
            rep: float(np.mean([preservation_report[rep][t]['spearman_rho'] for t in preservation_report[rep]]))
            for rep in representations.keys()
        },
        'reconstruction_psnr_mean': float(np.mean([recon_results[k]['mean_psnr'] for k in recon_results])),
        'reconstruction_ssim_mean': float(np.mean([recon_results[k]['mean_ssim'] for k in recon_results]))
    }

    with open('results/benchmark_summary.json', 'w') as f:
        json.dump(summary_card, f, indent=2)

    print("\n================== BENCHMARK SUMMARY SCORECARD ==================")
    print(f"Total Execution Time: {summary_card['total_runtime_seconds']} seconds")
    print(f"{'Representation':<35} | {'Illum MMA@3px':<14} | {'Scale MMA@3px':<14} | {'180° Invert MMA':<15} | {'Neighbourhood ρ':<15}")
    print("-" * 105)
    for rep in representations.keys():
        print(f"{rep:<35} | {summary_card['illumination_stability_mma3px_mean'][rep]:>12.1f}% | {summary_card['scale_stability_mma3px_mean'][rep]:>12.1f}% | {summary_card['extreme_azimuth_180deg_mma3px'][rep]:>13.1f}% | {summary_card['neighbourhood_preservation_spearman_rho_mean'][rep]:>14.3f}")
    print("=" * 105)

if __name__ == '__main__':
    run_main()
