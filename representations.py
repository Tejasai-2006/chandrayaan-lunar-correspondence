"""
representations.py
Implementations of:
1. Raw Intensity Correlation Baseline (ZNCC / normalized raw intensity vector) - MANDATORY BASELINE
2. Classical Phase & Gradient Invariant Representation (Multi-scale Gradient / Phase Congruency Proxy)
3. Lunar-InvNet: Self-Supervised Invariant Feature Encoder trained with Contrastive InfoNCE Loss
4. Lunar-AE: Invariant Autoencoder for reconstruction error and latent representation study
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import layers

class RawIntensityBaseline:
    """
    Mandatory Baseline: Raw Intensity Correlation.
    Flattens raw image patches, applies Zero-mean Normalization (ZNCC),
    and computes cosine similarity / Euclidean distance.
    """
    def __init__(self, patch_size=48):
        self.name = "Raw Intensity Correlation (Baseline)"
        self.patch_size = patch_size

    def extract_descriptors(self, patches):
        """
        patches: (N, H, W) or (N, H, W, 1) float32 in [0, 1]
        Returns: (N, D) normalized feature vectors
        """
        if patches is None or len(patches) == 0:
            return np.zeros((0, self.patch_size * self.patch_size), dtype=np.float32)

        if patches.ndim == 4:
            patches = patches[..., 0]
        N, H, W = patches.shape
        if (H, W) != (self.patch_size, self.patch_size):
            resized = []
            for i in range(N):
                r = cv2.resize(patches[i], (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
                resized.append(r)
            patches = np.stack(resized, axis=0)

        flat = patches.reshape(N, -1).astype(np.float32)
        # Zero-mean Normalization
        mean = np.mean(flat, axis=1, keepdims=True)
        std = np.std(flat, axis=1, keepdims=True) + 1e-7
        normed = (flat - mean) / std
        # L2 unit length
        l2_norm = np.linalg.norm(normed, axis=1, keepdims=True) + 1e-7
        return (normed / l2_norm).astype(np.float32)

class ClassicalGradientRepresentation:
    """
    Classical Handcrafted Invariant Representation:
    Multi-scale local gradient orientation histograms with local contrast normalization (SIFT / HOG proxy).
    """
    def __init__(self, patch_size=48, num_bins=8):
        self.name = "Multi-Scale Gradient Histogram"
        self.patch_size = patch_size
        self.num_bins = num_bins

    def extract_descriptors(self, patches):
        if patches is None or len(patches) == 0:
            return np.zeros((0, 16 * self.num_bins), dtype=np.float32)

        if patches.ndim == 4:
            patches = patches[..., 0]
        N, H, W = patches.shape
        descriptors = []
        for i in range(N):
            p = patches[i]
            if (H, W) != (self.patch_size, self.patch_size):
                p = cv2.resize(p, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)

            # Local Contrast Normalization (LCN)
            p_mean = cv2.GaussianBlur(p, (7, 7), 1.5)
            p_norm = p - p_mean

            # Gradients
            gx = cv2.Sobel(p_norm, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(p_norm, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
            angle = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)

            # 4x4 spatial cells
            cell_size = self.patch_size // 4
            hist_list = []
            bin_width = 2 * np.pi / self.num_bins

            for cy in range(4):
                for cx in range(4):
                    cell_mag = mag[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    cell_ang = angle[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                    hist = np.zeros(self.num_bins, dtype=np.float32)
                    for b in range(self.num_bins):
                        b_min = b * bin_width
                        b_max = (b + 1) * bin_width
                        mask = (cell_ang >= b_min) & (cell_ang < b_max)
                        hist[b] = np.sum(cell_mag[mask])
                    hist_list.append(hist)

            desc = np.concatenate(hist_list)
            desc_norm = desc / (np.linalg.norm(desc) + 1e-7)
            desc_clipped = np.clip(desc_norm, 0, 0.2)
            desc_final = desc_clipped / (np.linalg.norm(desc_clipped) + 1e-7)
            descriptors.append(desc_final)

        return np.array(descriptors, dtype=np.float32)

def build_lunar_invnet_backbone(input_shape=(48, 48, 1), embedding_dim=128):
    """
    Spatial Convolutional Encoder designed to eliminate illumination gain/bias
    while preserving spatial crater rim geometry and phase structure.
    """
    inputs = layers.Input(shape=input_shape)

    # Patch-level Group Normalization (acts as instance norm, removing illumination offset/gain)
    x = layers.GroupNormalization(groups=1)(inputs)

    # Conv Block 1
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 24x24

    # Conv Block 2
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 12x12

    # Conv Block 3
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  # 6x6

    # Preserve spatial crater rim configuration by flattening
    x = layers.Flatten()(x)  # 6 * 6 * 64 = 2304

    # Projection Head with Batch Normalization for zero-centered spherical representation
    proj = layers.Dense(256, activation='relu')(x)
    proj = layers.Dense(embedding_dim)(proj)
    proj_normed = layers.BatchNormalization()(proj)

    # L2 normalize onto unit hypersphere
    normed_emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name='l2_norm')(proj_normed)

    return keras.Model(inputs=inputs, outputs=normed_emb, name='LunarInvNet')

class LunarInvNetRepresentation:
    """Wrapper for trained Lunar-InvNet encoder."""
    def __init__(self, model, patch_size=48, embedding_dim=128):
        self.name = "Lunar-InvNet (Contrastive Invariant)"
        self.model = model
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

    def extract_descriptors(self, patches):
        if patches is None or len(patches) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        if patches.ndim == 3:
            patches = np.expand_dims(patches, axis=-1)
        N, H, W, C = patches.shape
        if (H, W) != (self.patch_size, self.patch_size):
            resized = []
            for i in range(N):
                r = cv2.resize(patches[i, ..., 0], (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
                resized.append(np.expand_dims(r, axis=-1))
            patches = np.stack(resized, axis=0)

        emb = self.model.predict(patches, batch_size=128, verbose=0)
        mean = np.mean(emb, axis=1, keepdims=True)
        centered = emb - mean
        norm = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-7
        return (centered / norm).astype(np.float32)

def build_lunar_autoencoder(input_shape=(48, 48, 1), latent_dim=64):
    """
    Convolutional Autoencoder for studying reconstruction error and latent invariance.
    """
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.GroupNormalization(groups=1)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)  # 24x24
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)  # 12x12
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x) # 6x6
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent_code')(x)

    # Decoder
    d = layers.Dense(6 * 6 * 128, activation='relu')(latent)
    d = layers.Reshape((6, 6, 128))(d)
    d = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(d) # 12x12
    d = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(d) # 24x24
    d = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(d) # 48x48
    decoded = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name='reconstruction')(d)

    autoencoder = keras.Model(inputs=inputs, outputs=decoded, name='LunarAutoencoder')
    encoder = keras.Model(inputs=inputs, outputs=latent, name='LunarEncoder')
    return autoencoder, encoder

class LunarAutoencoderRepresentation:
    """Wrapper for Autoencoder latent space representation."""
    def __init__(self, encoder, patch_size=48, latent_dim=64):
        self.name = "Lunar-AE (Latent Bottleneck)"
        self.encoder = encoder
        self.patch_size = patch_size
        self.latent_dim = latent_dim

    def extract_descriptors(self, patches):
        if patches is None or len(patches) == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)

        if patches.ndim == 3:
            patches = np.expand_dims(patches, axis=-1)
        N, H, W, C = patches.shape
        if (H, W) != (self.patch_size, self.patch_size):
            resized = []
            for i in range(N):
                r = cv2.resize(patches[i, ..., 0], (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
                resized.append(np.expand_dims(r, axis=-1))
            patches = np.stack(resized, axis=0)

        latent = self.encoder.predict(patches, batch_size=128, verbose=0)
        norm = np.linalg.norm(latent, axis=1, keepdims=True) + 1e-7
        return (latent / norm).astype(np.float32)
