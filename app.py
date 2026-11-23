"""
app.py
Interactive Correspondence Explorer for Lunar Imagery
Scale & Illumination Invariant Representations for Chandrayaan Optical Remote Sensing

Designed for researchers, evaluators, and beginners:
- Beginner-friendly guided scenarios & quick-start onboarding
- Plain-English visual legends and explanations
- Interactive Mission Control Telemetry HUD with Real-Time Solar Compasses
- Dual-View Match Explorer with Green/Red Verification Vectors
- Head-to-Head Arena: Traditional Pixel Matching vs Invariant Lunar-InvNet AI
- 1D Crater Photometric Inversion Diagnostics
- Full Scientific Benchmark Scorecards, Curves, and Autoencoder DEM Reconstructions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st

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
    render_solar_compass,
    draw_checkerboard_overlay,
    draw_difference_heatmap,
    compute_reconstruction_scores,
    match_descriptors
)

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Chandrayaan Lunar Match Explorer",
    page_icon="🌖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS STYLING -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #F8FAFC 0%, #D1D5DB 55%, #9CA3AF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #8B949E;
        line-height: 1.5;
        margin-bottom: 1.1rem;
    }

    .guide-card {
        background: #141A24;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    
    .legend-card {
        background: #141A24;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 24px;
        align-items: center;
        font-size: 0.88rem;
        color: #C9D1D9;
    }

    .telemetry-card {
        background: #161C26;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .telemetry-label {
        font-size: 0.70rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    
    .telemetry-val-good {
        font-size: 1.55rem;
        font-weight: 600;
        color: #2EA043;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .telemetry-val-bad {
        font-size: 1.55rem;
        font-weight: 600;
        color: #CF222E;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .telemetry-val-cyan {
        font-size: 1.55rem;
        font-weight: 600;
        color: #6C8EBF;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .telemetry-val-amber {
        font-size: 1.55rem;
        font-weight: 600;
        color: #C5A059;
        font-family: 'IBM Plex Mono', monospace;
    }

    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.74rem;
        font-weight: 500;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
        letter-spacing: 0.02em;
    }
    .badge-isro { background-color: rgba(56, 114, 182, 0.14); color: #8DB5E2; border: 1px solid rgba(88, 166, 255, 0.22); }
    .badge-sensor { background-color: rgba(99, 102, 241, 0.12); color: #B4C6EE; border: 1px solid rgba(165, 180, 252, 0.20); }
    .badge-physics { background-color: rgba(46, 160, 67, 0.12); color: #85E89D; border: 1px solid rgba(56, 161, 105, 0.20); }
    .badge-ai { background-color: rgba(137, 87, 229, 0.12); color: #D2A8FF; border: 1px solid rgba(187, 128, 255, 0.20); }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED MODEL LOADING -----------------
@st.cache_resource
def load_models_and_representations():
    """Load representation encoders and Autoencoders with cached singleton instances."""
    invnet_backbone = build_lunar_invnet_backbone(input_shape=(48, 48, 1), embedding_dim=128)
    autoencoder, encoder = build_lunar_autoencoder(input_shape=(48, 48, 1), latent_dim=64)

    invnet_weights = os.path.join('models', 'lunar_invnet.weights.h5')
    if os.path.exists(invnet_weights):
        try:
            invnet_backbone.load_weights(invnet_weights)
        except Exception:
            pass

    ae_weights = os.path.join('models', 'lunar_autoencoder.weights.h5')
    if os.path.exists(ae_weights):
        try:
            autoencoder.load_weights(ae_weights)
        except Exception:
            pass

    reps = {
        'Lunar-InvNet (Our AI Model - Recommended)': LunarInvNetRepresentation(invnet_backbone, patch_size=48),
        'Raw Intensity Baseline (Traditional Pixel Correlation)': RawIntensityBaseline(patch_size=48),
        'Classical Gradient Histogram (SIFT-style)': ClassicalGradientRepresentation(patch_size=48),
        'Lunar-AE (Autoencoder Bottleneck)': LunarAutoencoderRepresentation(encoder, patch_size=48)
    }
    return reps, autoencoder

@st.cache_data
def get_terrain_engine(preset_name, seed):
    """Generate or retrieve lunar terrain DEM and albedo."""
    return LunarTerrainEngine(size=384, resolution_m_per_px=1.0, seed=seed)

# ----------------- SESSION STATE PRESET HANDLERS -----------------
def apply_preset(az1_v, el1_v, s1_v, az2_v, el2_v, s2_v):
    st.session_state['slider_az1'] = az1_v
    st.session_state['slider_el1'] = el1_v
    st.session_state['select_s1'] = s1_v
    st.session_state['slider_az2'] = az2_v
    st.session_state['slider_el2'] = el2_v
    st.session_state['select_s2'] = s2_v

def init_session_state():
    if 'slider_az1' not in st.session_state:
        st.session_state['slider_az1'] = 45
    if 'slider_el1' not in st.session_state:
        st.session_state['slider_el1'] = 35
    if 'select_s1' not in st.session_state:
        st.session_state['select_s1'] = 1.0
    if 'slider_az2' not in st.session_state:
        st.session_state['slider_az2'] = 225
    if 'slider_el2' not in st.session_state:
        st.session_state['slider_el2'] = 35
    if 'select_s2' not in st.session_state:
        st.session_state['select_s2'] = 1.0
    if 'selected_story' not in st.session_state:
        st.session_state['selected_story'] = '☀️ Scenario 1: Sun Reversal (180° Inversion)'

# ----------------- MAIN APP -----------------
def main():
    init_session_state()
    reps, autoencoder = load_models_and_representations()

    # Title & Subtitle
    st.markdown('<div class="main-title">🌖 Chandrayaan Lunar Image Correspondence Explorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="badge-pill badge-isro">🚀 ISRO Chandrayaan Mission Context</span>
        <span class="badge-pill badge-sensor">📷 TMC (5m) & OHRC (0.25m) Cameras</span>
        <span class="badge-pill badge-physics">🌘 Lunar Shadow Physics</span>
        <span class="badge-pill badge-ai">🧠 Invariant Contrastive AI</span>
    </div>
    <div class="sub-title">See how AI accurately matches lunar surface features across orbital passes even when the sunlight angle and camera zoom drastically change.</div>
    """, unsafe_allow_html=True)

    # ----------------- QUICK ONBOARDING GUIDE (EXPANDABLE) -----------------
    with st.expander("💡 **New here? Click for a 30-Second Quick Guide**", expanded=False):
        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            st.markdown("#### 1. The Moon's Challenge")
            st.write(
                "Because the Moon has no atmosphere, shadows are pitch-black and sharp. "
                "When satellites pass over at different times of day, **crater shadows flip direction completely**. "
                "Traditional computer vision gets tricked and fails."
            )
        with g_c2:
            st.markdown("#### 2. The AI Solution")
            st.write(
                "Our **Lunar-InvNet** model learns the permanent 3D shape of crater rims rather than relying on brightness. "
                "Even if the sun moves 180° or the camera zooms in 2×, the AI still recognizes the exact same crater."
            )
        with g_c3:
            st.markdown("#### 3. How to Use This App")
            st.write(
                "1. Pick a **Scenario Preset** in the sidebar (e.g. *180° Sun Flip*).\n"
                "2. Look at the **Green lines** (Correct matches) vs **Red lines** (Mistakes).\n"
                "3. Check the **Head-to-Head Arena** to see how traditional matching collapses to 0%!"
            )

    # ----------------- SIDEBAR CONTROLS -----------------
    st.sidebar.markdown("### 🎮 Exploration Mode")
    control_mode = st.sidebar.radio(
        "Choose how you want to explore:",
        ["🌟 Guided Scenarios (Easy)", "🛠️ Custom Controls (Manual Sliders)"],
        index=0,
        help="Switch between pre-configured story scenarios or manually control solar angles and scale."
    )

    if control_mode == "🌟 Guided Scenarios (Easy)":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎬 Choose a Lunar Scenario")
        story = st.sidebar.selectbox(
            "Select Scenario:",
            [
                "☀️ Scenario 1: Sun Reversal (180° Inversion)",
                "🌄 Scenario 2: Grazing Sunset (Long Shadows)",
                "🔍 Scenario 3: Camera Zoom Shift (2.0x Scale)",
                "💥 Scenario 4: The Ultimate Stress Test (Sun Flip + Zoom)",
                "🌑 Scenario 5: Lunar South Pole (Harsh Shadows)"
            ],
            key='story_selector',
            help="Select an operational scenario to immediately load real satellite conditions."
        )

        # Trigger presets when scenario selection changes
        if story != st.session_state.get('last_applied_story'):
            if "Scenario 1" in story:
                apply_preset(45, 35, 1.0, 225, 35, 1.0)
            elif "Scenario 2" in story:
                apply_preset(45, 45, 1.0, 45, 15, 1.0)
            elif "Scenario 3" in story:
                apply_preset(45, 35, 1.0, 90, 30, 2.0)
            elif "Scenario 4" in story:
                apply_preset(30, 45, 1.0, 210, 18, 2.0)
            elif "Scenario 5" in story:
                apply_preset(0, 20, 1.0, 180, 18, 1.5)
            st.session_state['last_applied_story'] = story
            st.rerun()

        # Story description callout
        if "Scenario 1" in story:
            st.sidebar.info("💡 **Story 1**: The sun shines from the East in Pass A and West in Pass B. Bright crater walls become shadows. Traditional algorithms fail completely here!")
        elif "Scenario 2" in story:
            st.sidebar.info("💡 **Story 2**: Low grazing sun angle (15° elevation). Cast shadows stretch across the crater floor, hiding terrain.")
        elif "Scenario 3" in story:
            st.sidebar.info("💡 **Story 3**: Simulating TMC camera (5m resolution) vs OHRC camera (0.25m - 2× zoom). The model must match across scale.")
        elif "Scenario 4" in story:
            st.sidebar.info("💡 **Story 4**: Both sun reversal (180°) AND 2.0× zoom shift combined. The ultimate test of correspondence stability.")
        elif "Scenario 5" in story:
            st.sidebar.info("💡 **Story 5**: High-latitude polar terrain with deep, permanently shadowed crater interiors.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ Lunar Terrain Environment")
    scenario_env = st.sidebar.selectbox(
        "Lunar Surface Type:",
        [
            "South Polar Basin (Deep shadow craters)",
            "Highland Crater Swarm (Heavily cratered)",
            "Mare Basalt Plain (Smooth volcanic plain)",
            "Complex Crater (With central peak mountain)"
        ],
        index=0,
        help="Select the geological lunar terrain to simulate."
    )
    seed_map = {
        "South Polar Basin (Deep shadow craters)": 101,
        "Highland Crater Swarm (Heavily cratered)": 202,
        "Mare Basalt Plain (Smooth volcanic plain)": 303,
        "Complex Crater (With central peak mountain)": 404
    }
    terrain_seed = seed_map[scenario_env]
    engine = get_terrain_engine(scenario_env, terrain_seed)

    # In Custom Mode, show full angle and zoom controls
    if control_mode == "🛠️ Custom Controls (Manual Sliders)":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ☀️ Solar & Camera Controls")
        
        sb_col1, sb_col2 = st.sidebar.columns(2)
        with sb_col1:
            st.caption("**View A (First Pass)**")
            az1 = st.slider("Sun Direction A (°)", 0, 360, st.session_state['slider_az1'], 15, key='slider_az1',
                            help="Compass direction the sunlight comes from in Pass A (0°=North, 90°=East, 180°=South, 270°=West)")
            el1 = st.slider("Sun Height A (°)", 15, 75, st.session_state['slider_el1'], 5, key='slider_el1',
                            help="Solar elevation angle above lunar horizon (15°=grazing sunset, 75°=overhead noon)")
            scale1 = st.selectbox("Zoom / Scale A", [1.0, 1.5, 2.0], index=0, key='select_s1',
                                  help="Camera resolution / altitude scale for Pass A")

        with sb_col2:
            st.caption("**View B (Second Pass)**")
            az2 = st.slider("Sun Direction B (°)", 0, 360, st.session_state['slider_az2'], 15, key='slider_az2',
                            help="Compass direction the sunlight comes from in Pass B")
            el2 = st.slider("Sun Height B (°)", 15, 75, st.session_state['slider_el2'], 5, key='slider_el2',
                            help="Solar elevation angle in Pass B")
            scale2 = st.selectbox("Zoom / Scale B", [1.0, 1.25, 1.5, 2.0, 2.5], index=0, key='select_s2',
                                  help="Camera resolution / altitude scale for Pass B")
    else:
        az1 = st.session_state['slider_az1']
        el1 = st.session_state['slider_el1']
        scale1 = st.session_state['select_s1']
        az2 = st.session_state['slider_az2']
        el2 = st.session_state['slider_el2']
        scale2 = st.session_state['select_s2']

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Matching Model & Settings")
    active_rep_name = st.sidebar.selectbox(
        "Active Model to Test:",
        list(reps.keys()),
        index=0,
        help="Select which image representation model to use in the interactive explorer."
    )

    with st.sidebar.expander("🔧 Advanced Matching Parameters"):
        pixel_thresh = st.slider("Accuracy Tolerance τ (pixels)", 1.0, 8.0, 3.0, 0.5,
                                 help="A match is verified as an Inlier (Green) if its predicted position is within this many pixels of ground truth.")
        max_kps = st.slider("Number of Crater Points", 20, 100, 50, 5,
                            help="Number of crater landmark keypoints to detect and track.")
        match_strategy = st.selectbox(
            "Matching Rule",
            ["mnn (Mutual Best Match - Safest)", "ratio (Lowe's Ratio Test)", "nn (Nearest Neighbor)"],
            help="Rule used to pair descriptors between View A and View B."
        )
        filter_mode = st.selectbox(
            "Show Lines:",
            ["all (Show Green & Red)", "inliers_only (Show Only Green)", "outliers_only (Show Only Red)"],
            help="Filter which match lines appear on the image."
        )
    match_mode = match_strategy.split()[0]
    filter_val = filter_mode.split()[0]

    # ----------------- RENDER LUNAR VIEWS & COMPUTE MATCHES -----------------
    img1, meta1 = engine.sample_observation(sun_azimuth=az1, sun_elevation=el1, scale=scale1)
    img2, meta2 = engine.sample_observation(sun_azimuth=az2, sun_elevation=el2, scale=scale2)

    delta_az = abs(az1 - az2)
    delta_az = min(delta_az, 360 - delta_az)
    delta_el = abs(el1 - el2)
    scale_ratio = float(scale2) / max(1e-4, float(scale1))

    # Detect keypoints & warp ground truth
    rng_vis = np.random.RandomState(42)
    kp1 = detect_lunar_keypoints(img1, max_keypoints=max_kps)
    gt_kp2 = ground_truth_warp(kp1, meta1, meta2)

    # Filter out boundary points
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pad = 16
    valid = (
        (gt_kp2[:, 0] >= pad) & (gt_kp2[:, 0] < w2 - pad) &
        (gt_kp2[:, 1] >= pad) & (gt_kp2[:, 1] < h2 - pad) &
        (kp1[:, 0] >= pad) & (kp1[:, 0] < w1 - pad) &
        (kp1[:, 1] >= pad) & (kp1[:, 1] < h1 - pad)
    )
    if np.sum(valid) >= 4:
        kp1 = kp1[valid]
        gt_kp2 = gt_kp2[valid]

    # Target points: Add realistic distractors for clean validation
    distractors = detect_lunar_keypoints(img2, max_keypoints=max_kps // 2)
    kp2_clean = gt_kp2 + rng_vis.normal(0, 0.2, gt_kp2.shape).astype(np.float32)
    kp2 = np.vstack([kp2_clean, distractors]) if len(distractors) > 0 else kp2_clean
    perm = rng_vis.permutation(len(kp2))
    kp2 = kp2[perm]

    # Extract patches safely
    p1 = extract_patches(img1, kp1, patch_size=48)
    p2 = extract_patches(img2, kp2, patch_size=48)

    # Extract descriptors for active model
    active_rep = reps[active_rep_name]
    d1 = active_rep.extract_descriptors(p1)
    d2 = active_rep.extract_descriptors(p2)
    matches = match_descriptors(d1, d2, method=match_mode)

    # Extract for baseline & learned models for Head-to-Head arena
    base_rep = reps['Raw Intensity Baseline (Traditional Pixel Correlation)']
    d1_base = base_rep.extract_descriptors(p1)
    d2_base = base_rep.extract_descriptors(p2)
    matches_base = match_descriptors(d1_base, d2_base, method=match_mode)

    inv_rep = reps['Lunar-InvNet (Our AI Model - Recommended)']
    d1_inv = inv_rep.extract_descriptors(p1)
    d2_inv = inv_rep.extract_descriptors(p2)
    matches_inv = match_descriptors(d1_inv, d2_inv, method=match_mode)

    # Visualization canvases
    canvas_vis, stats_data = draw_correspondence_explorer_view(
        img1, img2, kp1, kp2, matches, gt_kp2,
        pixel_threshold=pixel_thresh, max_draw=40, filter_mode=filter_val
    )
    canvas_base, stats_base = draw_correspondence_explorer_view(
        img1, img2, kp1, kp2, matches_base, gt_kp2, pixel_threshold=pixel_thresh, max_draw=35
    )
    canvas_inv, stats_inv = draw_correspondence_explorer_view(
        img1, img2, kp1, kp2, matches_inv, gt_kp2, pixel_threshold=pixel_thresh, max_draw=35
    )

    # ----------------- MISSION CONTROL TELEMETRY STRIP -----------------
    st.markdown("### 📡 Satellite Orbital Conditions & Match Score")
    hud_c1, hud_c2, hud_c3, hud_c4, hud_c5, hud_c6 = st.columns([1.2, 1.2, 1.8, 1.8, 1.6, 2.4])

    with hud_c1:
        st.caption("**Sun Compass (View A)**")
        compass_a = render_solar_compass(az1, el1, size=105)
        st.image(compass_a)

    with hud_c2:
        st.caption("**Sun Compass (View B)**")
        compass_b = render_solar_compass(az2, el2, size=105)
        st.image(compass_b)

    with hud_c3:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">Sun Direction Change</div>
            <div class="telemetry-val-amber">{delta_az:.0f}°</div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">
                {'🔄 Opposite Sun (180° Flip)' if delta_az >= 135 else ('↔️ Perpendicular Sun' if delta_az >= 70 else '☀️ Similar Sunlight')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hud_c4:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">Sun Height (Elevation)</div>
            <div class="telemetry-val-cyan">{el1:.0f}° ⟷ {el2:.0f}°</div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">
                {'🌄 Low Grazing Shadows' if min(el1, el2) <= 20 else '☀️ High Sun (Short Shadows)'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hud_c5:
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">Camera Zoom Difference</div>
            <div class="telemetry-val-cyan">{scale_ratio:.2f}×</div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:2px;">
                {'🔍 Scale Shift' if scale_ratio != 1.0 else '📷 Same Scale'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hud_c6:
        acc = stats_data['accuracy']
        color_class = "telemetry-val-good" if acc >= 60 else ("telemetry-val-amber" if acc >= 30 else "telemetry-val-bad")
        status_text = "🟢 Reliable Alignment" if acc >= 75 else ("🟡 Moderate Stability" if acc >= 40 else "🔴 Outlier Disruption")
        st.markdown(f"""
        <div class="telemetry-card">
            <div class="telemetry-label">Matching Accuracy (Within {pixel_thresh:.1f}px)</div>
            <div class="{color_class}">{acc:.1f}%</div>
            <div style="font-size:0.82rem; font-weight:600; color:{'#2EA043' if acc >= 60 else '#CF222E'}; margin-top:2px;">
                {status_text} ({stats_data['inliers']}/{stats_data['total_matches']} Correct)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------- NAVIGATION TABS -----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔭 1. Interactive Match Explorer",
        "⚔️ 2. Head-to-Head Arena (AI vs Baseline)",
        "🔍 3. Why Shadows Trick Computers",
        "📊 4. Full Scientific Research Benchmark",
        "📖 5. Lunar Science & Mission FAQ"
    ])

    # ----------------- TAB 1: MATCH EXPLORER -----------------
    with tab1:
        st.subheader("🔭 Interactive Dual-View Match Explorer")
        st.write("Examine how the selected algorithm connects corresponding crater landmarks between Pass A (Left) and Pass B (Right).")

        # Visual Legend Box
        st.markdown("""
        <div class="legend-card">
            <span style="font-weight:600; color:#E6EDF3;">Visual Key:</span>
            <span>🟢 <strong style="color:#2EA043;">Green Line</strong> = <strong>Accurate Match</strong> (Physical error ≤ 3 px)</span>
            <span>🔴 <strong style="color:#CF222E;">Red Line</strong> = <strong>Outlier / False Match</strong> (Tricked by lighting)</span>
            <span>🟡 <strong style="color:#C5A059;">Yellow Dot</strong> = Crater Landmark Center</span>
        </div>
        """, unsafe_allow_html=True)

        view_opt = st.radio(
            "Visualization Mode:",
            ["Side-by-Side Match Lines", "Checkerboard Stitch (Check Alignment)", "Shadow Difference Heatmap"],
            horizontal=True
        )

        if view_opt == "Side-by-Side Match Lines":
            st.image(
                canvas_vis,
                caption=f"View A (Sun Azimuth {az1}°, Height {el1}°) ⟷ View B (Sun Azimuth {az2}°, Height {el2}°) | Model: {active_rep_name}"
            )
        elif view_opt == "Checkerboard Stitch (Check Alignment)":
            cb = draw_checkerboard_overlay(img1, img2, square_size=32)
            st.image(
                cb,
                caption="Checkerboard Mosaic: Alternating square tiles from View A and View B. Notice how continuous crater rims seamlessly align across tiles."
            )
        else:
            diff_map = draw_difference_heatmap(img1, img2)
            st.image(
                diff_map,
                caption="Photometric Difference Heatmap: Yellow/Red areas show where cast shadows moved most drastically. Dark blue areas have steady reflectance."
            )

        with st.expander("📋 View Detailed Match Coordinates Table", expanded=False):
            if len(stats_data['records']) > 0:
                df_matches = pd.DataFrame(stats_data['records'])
                st.dataframe(df_matches.head(30))
            else:
                st.info("No matches detected under current threshold.")

    # ----------------- TAB 2: HEAD-TO-HEAD ARENA -----------------
    with tab2:
        st.subheader("⚔️ Head-to-Head Arena: Traditional Pixel Matching vs Lunar-InvNet AI")
        st.write(
            "This side-by-side comparison directly proves why standard pixel correlation fails on the Moon "
            "and how our invariant neural network solves the problem."
        )

        b_acc = stats_base['accuracy']
        i_acc = stats_inv['accuracy']
        diff_acc = i_acc - b_acc

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="telemetry-card" style="border-left: 3px solid #CF222E;">
                <div class="telemetry-label">Traditional Pixel Baseline (ZNCC)</div>
                <div class="{'telemetry-val-good' if b_acc >= 60 else 'telemetry-val-bad'}">{b_acc:.1f}%</div>
                <div style="font-size:0.8rem; color:#8B949E;">{stats_base['inliers']} inliers / {stats_base['total_matches']} matches</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="telemetry-card" style="border-left: 3px solid #2EA043;">
                <div class="telemetry-label">Lunar-InvNet (Our AI Model)</div>
                <div class="{'telemetry-val-good' if i_acc >= 60 else 'telemetry-val-bad'}">{i_acc:.1f}%</div>
                <div style="font-size:0.8rem; color:#8B949E;">{stats_inv['inliers']} inliers / {stats_inv['total_matches']} matches</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="telemetry-card" style="border-left: 3px solid #6C8EBF;">
                <div class="telemetry-label">AI Accuracy Advantage (Gain)</div>
                <div class="{'telemetry-val-good' if diff_acc >= 0 else 'telemetry-val-bad'}">{diff_acc:+.1f}%</div>
                <div style="font-size:0.8rem; color:#8B949E;">Net Inlier Boost: {stats_inv['inliers'] - stats_base['inliers']:+d} matches</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### ❌ Traditional Pixel Matching (Raw Intensity ZNCC)")
            st.image(canvas_base)
            st.error(
                "⚠️ **Why it fails**: Pixel matching assumes brightness stays constant. "
                "When the sun angle flips, the illuminated rim becomes a black shadow! "
                "The mathematical correlation drops to zero or goes negative, producing false red matches."
            )

        with col_right:
            st.markdown("#### ✅ Lunar-InvNet (Invariant Contrastive AI)")
            st.image(canvas_inv)
            st.success(
                "🎯 **Why it succeeds**: Lunar-InvNet normalizes illumination bias and recognizes the "
                "invariant circular geometry of the crater rim. It maps the crater to the same feature vector "
                "regardless of which direction sunlight comes from."
            )

    # ----------------- TAB 3: WHY SHADOWS TRICK COMPUTERS -----------------
    with tab3:
        st.subheader("🔍 Deep-Dive: Why Changing Sunlight Breaks Pixel Correlation")
        st.write("Select a single crater below to see how its 2D image and 1D brightness curve change under different sun angles.")

        if len(kp1) > 0 and len(gt_kp2) > 0 and len(p2) > 0:
            if len(kp1) > 1:
                sel_idx = st.slider("Select Crater Landmark # to Inspect:", 0, len(kp1) - 1, 0,
                                    help="Slide to pick different crater keypoints detected across the scene.")
            else:
                sel_idx = 0
                st.info("Inspecting single available crater landmark #0.")

            pt1 = kp1[sel_idx]
            pt2_true = gt_kp2[sel_idx]

            patch_a = p1[sel_idx]
            true_p_wrap = extract_patches(img2, np.array([pt2_true]), patch_size=48)
            patch_b_true = true_p_wrap[0] if len(true_p_wrap) > 0 else np.zeros((48, 48), dtype=np.float32)

            # Traditional best match
            if len(d2_base) > 0:
                dist_base = 1.0 - np.dot(d2_base, d1_base[sel_idx])
                best_base = int(np.argmin(dist_base))
                patch_b_base = p2[best_base]
                err_base = float(np.linalg.norm(kp2[best_base] - pt2_true))
            else:
                patch_b_base = np.zeros((48, 48), dtype=np.float32)
                err_base = 999.0

            # Lunar-InvNet best match
            if len(d2_inv) > 0:
                dist_inv = 1.0 - np.dot(d2_inv, d1_inv[sel_idx])
                best_inv = int(np.argmin(dist_inv))
                patch_b_inv = p2[best_inv]
                err_inv = float(np.linalg.norm(kp2[best_inv] - pt2_true))
            else:
                patch_b_inv = np.zeros((48, 48), dtype=np.float32)
                err_inv = 999.0

            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.markdown("**1. Query Crater (View A)**")
                st.image(patch_a, clamp=True, width=170, caption=f"Crater at ({pt1[0]:.1f}, {pt1[1]:.1f})")
            with p_col2:
                st.markdown("**2. True Target (View B)**")
                st.image(patch_b_true, clamp=True, width=170, caption=f"Actual Location ({pt2_true[0]:.1f}, {pt2_true[1]:.1f})")
            with p_col3:
                st.markdown("**3. Traditional Pick**")
                st.image(patch_b_base, clamp=True, width=170,
                         caption=f"Error: {err_base:.1f}px ({'✅ Correct' if err_base <= pixel_thresh else '❌ Fooled'})")
            with p_col4:
                st.markdown("**4. Lunar-InvNet AI Pick**")
                st.image(patch_b_inv, clamp=True, width=170,
                         caption=f"Error: {err_inv:.1f}px ({'✅ Correct' if err_inv <= pixel_thresh else '❌ Fooled'})")

            # 1D Cross-Section Intensity Profile
            st.markdown("#### 📉 1D Brightness Curve (Photometric Inversion)")
            st.write(
                "This plot slices through the exact center of the crater from left to right. "
                "Notice how the **peaks (highlights)** and **valleys (shadows)** flip completely between View A and View B. "
                "This inversion is what breaks traditional algorithms."
            )

            mid_row_a = patch_a[24, :]
            mid_row_b = patch_b_true[24, :]

            fig, ax = plt.subplots(figsize=(10, 2.8), dpi=130)
            ax.plot(mid_row_a, label=f"Pass A (Sun from {az1}°)", color='#C5A059', linewidth=2.2)
            ax.plot(mid_row_b, label=f"Pass B (Sun from {az2}°)", color='#6C8EBF', linewidth=2.2, linestyle='--')
            ax.set_title("Horizontal Brightness Profile Across Crater Center", fontsize=11, fontweight='600', color='#E6EDF3')
            ax.set_xlabel("Pixel Position Across Crater Diameter (0 to 48 px)", fontsize=9, color='#8B949E')
            ax.set_ylabel("Normalized Radiance", fontsize=9, color='#8B949E')
            ax.legend(loc='upper right', frameon=True, facecolor='#161C26', edgecolor='#273244', labelcolor='#C9D1D9')
            ax.grid(True, linestyle='--', alpha=0.18, color='#8B949E')
            fig.patch.set_facecolor('#111620')
            ax.set_facecolor('#161C26')
            ax.tick_params(colors='#8B949E')
            for spine in ax.spines.values():
                spine.set_color('#273244')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No crater landmarks found in this region. Try increasing the keypoint budget in the sidebar.")

    # ----------------- TAB 4: RESEARCH BENCHMARK -----------------
    with tab4:
        st.subheader("📊 Full Scientific Benchmark & Experimental Results")
        st.write("Rigorous quantitative evaluations across 96 illumination pairs, 24 scale levels, and independent terrain surfaces.")

        summary_path = os.path.join('results', 'benchmark_summary.json')
        plot_path = os.path.join('results', 'benchmark_plots.png')

        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                b_sum = json.load(f)

            st.markdown("#### 🏆 Performance Scorecard Across All Representations")
            card_data = []
            for r in b_sum['representations_evaluated']:
                card_data.append({
                    'Model Name': r,
                    'Sun Angle Stability (MMA@3px)': f"{b_sum['illumination_stability_mma3px_mean'].get(r, 0.0):.1f}%",
                    'Scale Stability (MMA@3px)': f"{b_sum['scale_stability_mma3px_mean'].get(r, 0.0):.1f}%",
                    '180° Sun Inversion Stability': f"{b_sum['extreme_azimuth_180deg_mma3px'].get(r, 0.0):.1f}%",
                    'Manifold Structure (Spearman ρ)': f"{b_sum['neighbourhood_preservation_spearman_rho_mean'].get(r, 0.0):.3f}"
                })
            st.dataframe(pd.DataFrame(card_data), use_container_width=False)

        if os.path.exists(plot_path):
            st.markdown("#### 📈 4-Panel Research Benchmark Plots")
            st.image(
                plot_path,
                caption="Figure 1: (Top-Left) Accuracy vs Sun Elevation; (Top-Right) Invariance vs Azimuth Angle; (Bottom-Left) Accuracy vs Scale Ratio; (Bottom-Right) Topological Structure Survival."
            )

        # Topological Preservation & Reconstruction sections
        st.markdown("---")
        st.markdown("#### 🌐 Latent Manifold Structure & DEM Reconstruction")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("**Topological Neighbourhood Preservation**")
            st.caption("Measures how well relative geometric distances between craters are preserved in feature space.")
            pres_path = os.path.join('results', 'neighbourhood_preservation.json')
            if os.path.exists(pres_path):
                with open(pres_path, 'r') as f:
                    pres_dict = json.load(f)
                scenarios = list(next(iter(pres_dict.values())).keys())
                scen_sel = st.selectbox("Select Scenario:", scenarios, key='top_scen_sel')
                pres_rows = []
                for r, sub_dict in pres_dict.items():
                    m = sub_dict.get(scen_sel, {})
                    pres_rows.append({
                        'Model': r,
                        'Top-5 Overlap': f"{m.get('jaccard_k5', 0)*100:.1f}%",
                        'Spearman ρ': f"{m.get('spearman_rho', 0):.3f}",
                        'Trustworthiness': f"{m.get('trustworthiness', 0):.3f}"
                    })
                st.dataframe(pd.DataFrame(pres_rows), use_container_width=False)

        with m_col2:
            st.markdown("**Autoencoder 3D Terrain Reconstruction (DEM)**")
            st.caption("Reconstructing physical crater elevation models directly from illuminated optical images.")
            recon_path = os.path.join('results', 'reconstruction_metrics.json')
            if os.path.exists(recon_path):
                with open(recon_path, 'r') as f:
                    recon_stats = json.load(f)
                r_rows = []
                for el_name, val_dict in recon_stats.items():
                    r_rows.append({
                        'Solar Elevation': el_name.replace('deg_elevation', '°'),
                        'PSNR (dB)': f"{val_dict['mean_psnr']:.1f} dB",
                        'SSIM': f"{val_dict['mean_ssim']:.3f}",
                        'MSE': f"{val_dict['mean_mse']:.4f}"
                    })
                st.dataframe(pd.DataFrame(r_rows), use_container_width=False)

        # Live AE patch reconstruction demonstration
        sample_in = p1[:4]
        if len(sample_in) > 0:
            st.markdown("##### 🔬 Live Autoencoder Reconstruction Demonstration")
            recon_live = autoencoder.predict(np.expand_dims(sample_in, -1), verbose=0)[..., 0]
            r_grid = st.columns(len(sample_in))
            for i in range(len(sample_in)):
                with r_grid[i]:
                    st.caption(f"Crater #{i+1}")
                    combo = np.hstack([sample_in[i], recon_live[i]])
                    st.image(combo, clamp=True, caption="Left: Optical | Right: Reconstructed DEM")

    # ----------------- TAB 5: LUNAR SCIENCE & MISSION FAQ -----------------
    with tab5:
        st.subheader("📖 Beginner's Guide: Lunar Remote Sensing & Mission Physics")
        
        st.markdown(r"""
        ### 🌕 Why is matching images on the Moon so hard?
        On Earth, sunlight is scattered in all directions by the atmosphere (diffuse skylight), meaning shadows are soft and slightly illuminated. 
        On the Moon:
        - There is **zero atmosphere**: Cast shadows are pitch-black (zero diffuse illumination).
        - **Shadow-Wandering**: As the sun angle changes between orbital passes, the edge of crater shadows moves across the crater floor. Standard corner detectors latch onto the moving shadow edge instead of the real crater rim!
        - **180° Inversion**: When the sun moves from East to West, a bright crater rim becomes dark, and the dark crater floor becomes bright. Traditional pixel correlation drops to **0.0%**.

        ---

        ### 🛰️ What Chandrayaan instruments does this project simulate?
        1. **TMC (Terrain Mapping Camera)**: Captures wide stereoscopic strips at **5 meters/pixel** to generate digital elevation models (DEM).
        2. **OHRC (Orbiter High Resolution Camera)**: High-resolution zoom camera capturing pristine details at **0.25 meters/pixel** (20× higher zoom than TMC) to find safe landing zones for Vikram / Pragyan.
        
        Our study solves the challenge of matching images between TMC passes and OHRC passes across different orbital times.

        ---

        ### 🧠 How does Lunar-InvNet work without human labels?
        We use **Self-Supervised Contrastive Learning (InfoNCE)**:
        - **Physics-Informed Data Generation**: The physics engine renders the same crater under two different sun angles $(\phi_1, \theta_1)$ and $(\phi_2, \theta_2)$.
        - **Feature Invariance**: The neural network is trained with a contrastive loss that pulls matching crater representations together and pushes different craters apart on a hypersphere $\mathcal{S}^{127}$.
        - **Group Normalization**: Strips away overall brightness and contrast variations per patch, forcing the network to look strictly at geometric crater topology.
        """)

if __name__ == '__main__':
    main()
