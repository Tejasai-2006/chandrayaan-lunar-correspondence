"""
utils.py
Helper utilities for:
- Keypoint detection across lunar terrain features (crater rims, ridges, boulders)
- Patch extraction with reflection/border padding
- Mutual Nearest Neighbor (MNN) and nearest-neighbor matching
- Exact ground-truth coordinate mapping between multi-scale, multi-illumination views
- Visualizing correspondence lines, error maps, and side-by-side comparisons
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def detect_lunar_keypoints(image, max_keypoints=120, quality_level=0.012, min_distance=12):
    """
    Detect stable interest points on lunar terrain (crater rims, ejecta, topographical relief).
    Uses Shi-Tomasi / Harris corner detection with adaptive margins and safe fallback.
    """
    img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    h, w = img_u8.shape[:2]
    margin = max(12, min(24, int(min(h, w) * 0.1)))

    mask = np.zeros_like(img_u8)
    if h > 2 * margin and w > 2 * margin:
        mask[margin:-margin, margin:-margin] = 255
    else:
        mask[:] = 255

    corners = cv2.goodFeaturesToTrack(
        img_u8,
        maxCorners=max_keypoints,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        useHarrisDetector=True,
        k=0.04
    )

    if corners is None or len(corners) < 4:
        # Guaranteed regular grid fallback inside margins
        step = max(16, int(min(h, w) / 8))
        ys = np.arange(margin, h - margin, step)
        xs = np.arange(margin, w - margin, step)
        if len(ys) == 0 or len(xs) == 0:
            ys = np.array([h // 2], dtype=np.int32)
            xs = np.array([w // 2], dtype=np.int32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
        kps = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        return kps[:max_keypoints]

    return corners.reshape(-1, 2).astype(np.float32)

def extract_patches(image, keypoints, patch_size=48):
    """
    Extract square patches centered at keypoint coordinates (x, y).
    Uses cv2.getRectSubPix for sub-pixel accuracy and robust boundary handling.
    Always returns shape (N, patch_size, patch_size) float32.
    """
    if keypoints is None or len(keypoints) == 0:
        return np.zeros((0, patch_size, patch_size), dtype=np.float32)

    img_f32 = np.clip(image, 0.0, 1.0).astype(np.float32)
    patches = []
    for pt in keypoints:
        center = (float(pt[0]), float(pt[1]))
        patch = cv2.getRectSubPix(img_f32, (patch_size, patch_size), center)
        patches.append(patch.astype(np.float32))

    return np.array(patches, dtype=np.float32)

def ground_truth_warp(points_1, meta_1, meta_2):
    """
    Maps keypoint coordinates from image 1 to image 2 using exact DEM projection metadata.
    points_1: (N, 2) in image 1 coordinates [x1, y1]
    meta_1, meta_2: metadata dicts containing 'x0', 'y0', 'scale'
    Returns: (N, 2) exact ground truth coordinates in image 2 [x2, y2]
    """
    if points_1 is None or len(points_1) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    s1 = meta_1.get('scale', 1.0)
    s2 = meta_2.get('scale', 1.0)
    x0_1, y0_1 = meta_1.get('x0', 0), meta_1.get('y0', 0)
    x0_2, y0_2 = meta_2.get('x0', 0), meta_2.get('y0', 0)

    x_dem = points_1[:, 0] * s1 + x0_1
    y_dem = points_1[:, 1] * s1 + y0_1

    x2 = (x_dem - x0_2) / max(1e-4, s2)
    y2 = (y_dem - y0_2) / max(1e-4, s2)

    return np.column_stack([x2, y2]).astype(np.float32)

def match_descriptors(desc1, desc2, method='mnn', ratio_threshold=0.88):
    """
    Match descriptors between two sets with complete boundary safety.
    method: 'mnn' (Mutual Nearest Neighbors) or 'nn' (Standard Nearest Neighbor) or 'ratio' (Lowe's ratio)
    Returns: list of (idx1, idx2, distance)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    # Ensure descriptors are 2D
    if desc1.ndim == 1:
        desc1 = np.expand_dims(desc1, 0)
    if desc2.ndim == 1:
        desc2 = np.expand_dims(desc2, 0)

    # Cosine distance in [0, 2]
    dot = np.dot(desc1, desc2.T)
    dist_matrix = 1.0 - np.clip(dot, -1.0, 1.0)

    n1, n2 = dist_matrix.shape
    if n1 == 0 or n2 == 0:
        return []

    matches = []
    if method == 'mnn':
        best_2_for_1 = np.argmin(dist_matrix, axis=1)
        best_1_for_2 = np.argmin(dist_matrix, axis=0)

        for i, j in enumerate(best_2_for_1):
            if best_1_for_2[j] == i:
                d = dist_matrix[i, j]
                matches.append((int(i), int(j), float(d)))

    elif method == 'ratio':
        for i in range(n1):
            sorted_indices = np.argsort(dist_matrix[i])
            d1 = dist_matrix[i, sorted_indices[0]]
            d2 = dist_matrix[i, sorted_indices[1]] if len(sorted_indices) > 1 else 1e5
            if d1 / (d2 + 1e-7) < ratio_threshold:
                matches.append((int(i), int(sorted_indices[0]), float(d1)))

    else: # Standard nearest neighbor
        best_2_for_1 = np.argmin(dist_matrix, axis=1)
        for i, j in enumerate(best_2_for_1):
            matches.append((int(i), int(j), float(dist_matrix[i, j])))

    return matches

def render_solar_compass(azimuth_deg, elevation_deg, size=140):
    """
    Renders an orbital telemetry solar vector compass HUD widget.
    Returns RGB uint8 image.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8) + 16
    center = (size // 2, size // 2)
    radius = size // 2 - 14

    # Outer and inner rings
    cv2.circle(canvas, center, radius, (55, 65, 85), 1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius // 2, (40, 50, 70), 1, cv2.LINE_AA)

    # Cardinal axes
    cv2.line(canvas, (center[0], 10), (center[0], size - 10), (35, 45, 65), 1)
    cv2.line(canvas, (10, center[1]), (size - 10, center[1]), (35, 45, 65), 1)

    # Cardinal text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "N", (center[0] - 5, 20), font, 0.35, (140, 160, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "S", (center[0] - 4, size - 8), font, 0.35, (140, 160, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "E", (size - 16, center[1] + 4), font, 0.35, (140, 160, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "W", (6, center[1] + 4), font, 0.35, (140, 160, 190), 1, cv2.LINE_AA)

    # Solar vector (azimuth clockwise from top, elevation scales radial distance)
    rad = np.radians(azimuth_deg)
    dist = int(radius * np.cos(np.radians(elevation_deg)))
    sx = int(center[0] + dist * np.sin(rad))
    sy = int(center[1] - dist * np.cos(rad))

    # Glow line and Sun icon
    cv2.line(canvas, center, (sx, sy), (0, 190, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (sx, sy), 8, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (sx, sy), 11, (255, 255, 255), 1, cv2.LINE_AA)

    # Center nadir observer dot
    cv2.circle(canvas, center, 3, (200, 220, 255), -1, cv2.LINE_AA)

    # Convert to RGB for Streamlit display
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def draw_checkerboard_overlay(img1, img2, square_size=28):
    """Creates a 3-channel RGB checkerboard pattern alternating between img1 and img2."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h, target_w = min(h1, h2), min(w1, w2)
    i1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
    i2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    for y in range(0, target_h, square_size):
        for x in range(0, target_w, square_size):
            y_end = min(y + square_size, target_h)
            x_end = min(x + square_size, target_w)
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                canvas[y:y_end, x:x_end] = i1[y:y_end, x:x_end]
            else:
                canvas[y:y_end, x:x_end] = i2[y:y_end, x:x_end]
    c_u8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(c_u8, cv2.COLOR_GRAY2RGB)

def draw_difference_heatmap(img1, img2):
    """Computes absolute photometric difference heatmap in pseudo-color RGB."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h, target_w = min(h1, h2), min(w1, w2)
    i1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
    i2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
    diff = np.abs(i1 - i2)
    diff_u8 = (np.clip(diff / (np.max(diff) + 1e-6), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def draw_correspondence_explorer_view(
    img1, img2, kp1, kp2, matches, gt_kp2, pixel_threshold=3.0, max_draw=45,
    filter_mode='all', highlight_idx=None, line_thickness=1
):
    """
    Render side-by-side correspondence plot with color-coded matches:
    - Green line: Inlier match (error <= pixel_threshold)
    - Red line: Outlier match (error > pixel_threshold)
    - Highlighted selection: Cyan glow
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas_h = max(h1, h2)
    canvas_w = w1 + w2 + 18
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Convert grayscale to RGB
    img1_rgb = cv2.cvtColor((np.clip(img1, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor((np.clip(img2, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    canvas[:h1, :w1] = img1_rgb
    canvas[:h2, w1 + 18:canvas_w] = img2_rgb

    # Sleek divider bar
    canvas[:, w1:w1+18] = [22, 27, 34]
    cv2.line(canvas, (w1 + 9, 0), (w1 + 9, canvas_h), (50, 65, 85), 1)

    offset_x = w1 + 18
    inliers = 0
    outliers = 0
    draw_count = 0

    if matches is None or len(matches) == 0:
        return canvas, {
            'total_matches': 0, 'inliers': 0, 'outliers': 0,
            'accuracy': 0.0, 'threshold': pixel_threshold, 'records': []
        }

    sorted_matches = sorted(matches, key=lambda m: m[2])
    match_records = []

    for idx1, idx2, dist in sorted_matches:
        if idx1 >= len(kp1) or idx2 >= len(kp2) or idx1 >= len(gt_kp2):
            continue

        p1 = kp1[idx1]
        p2 = kp2[idx2]
        true_p2 = gt_kp2[idx1]

        error = float(np.linalg.norm(p2 - true_p2))
        is_inlier = error <= pixel_threshold

        if is_inlier:
            inliers += 1
            color = (34, 197, 94)  # Vibrant Emerald
        else:
            outliers += 1
            color = (239, 68, 68)  # Crimson Red

        match_records.append({
            'idx1': idx1, 'idx2': idx2, 'dist': dist,
            'error': error, 'is_inlier': is_inlier
        })

        if filter_mode == 'inliers_only' and not is_inlier:
            continue
        if filter_mode == 'outliers_only' and is_inlier:
            continue

        if draw_count < max_draw:
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0] + offset_x)), int(round(p2[1])))

            if highlight_idx is not None and idx1 == highlight_idx:
                hl_color = (6, 182, 212) # Electric Cyan
                cv2.circle(canvas, pt1, 7, hl_color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pt2, 7, hl_color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pt1, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt2, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, hl_color, line_thickness + 2, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt1, 4, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt2, 4, color, -1, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

            draw_count += 1

    total = inliers + outliers
    accuracy = (inliers / total * 100.0) if total > 0 else 0.0

    return canvas, {
        'total_matches': total,
        'inliers': inliers,
        'outliers': outliers,
        'accuracy': accuracy,
        'threshold': pixel_threshold,
        'records': match_records
    }

def compute_reconstruction_scores(orig, recon):
    """Compute MSE, PSNR, and SSIM between original and reconstructed patches/images."""
    orig_c = np.clip(orig, 0.0, 1.0).astype(np.float32)
    recon_c = np.clip(recon, 0.0, 1.0).astype(np.float32)

    mse = float(np.mean((orig_c - recon_c)**2))
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-10)))
    data_range = 1.0
    ssim_val = float(ssim(orig_c, recon_c, data_range=data_range))
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim_val}
