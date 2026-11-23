"""
lunar_engine.py
Physics-based Lunar Terrain Generator and Photometric Renderer.
Simulates Chandrayaan TMC / OHRC orbital imaging conditions:
- Crater geomorphology (Pike's crater morphometrics, power-law distributions, central peaks, degradation)
- Fractal regolith terrain (fractional Brownian motion)
- Lommel-Seeliger & Lunar-Lambert photometric reflectance
- Horizon ray-traced cast shadows
- Scale changes, camera resolution sampling, and sub-pixel ground truth correspondences
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class LunarTerrainEngine:
    def __init__(self, size=512, resolution_m_per_px=1.0, seed=42):
        self.size = size
        self.resolution = resolution_m_per_px
        self.rng = np.random.RandomState(seed)
        self.dem = None
        self.albedo = None
        self.generate_terrain()

    def generate_terrain(self):
        """Generate high-fidelity lunar DEM with craters and fractal roughness."""
        size = self.size
        # 1. Fractal background terrain (lunar mare / highland topography)
        dem = self._fractal_brownian_noise(size, octaves=6, persistence=0.55, lacunarity=2.0) * 120.0
        albedo = np.full((size, size), 0.12, dtype=np.float32)  # Baseline mature regolith albedo

        # 2. Add realistic impact craters following lunar power-law distribution
        # Number of craters varies inversely with diameter squared: N(D) ~ D^-2
        num_craters = 45
        min_rad = 6
        max_rad = size // 5

        # Pareto distribution for crater radii
        radii = (min_rad + (max_rad - min_rad) * (self.rng.pareto(a=1.8, size=num_craters * 3) / 4.0))
        radii = np.clip(radii, min_rad, max_rad)
        radii = np.sort(radii)[::-1][:num_craters]

        for r in radii:
            cx = self.rng.randint(int(r), size - int(r))
            cy = self.rng.randint(int(r), size - int(r))
            dem, albedo = self._add_crater(dem, albedo, cx, cy, r)

        # Smooth slightly to mimic micro-meteorite bombardment / regolith gardening
        self.dem = gaussian_filter(dem, sigma=0.8)
        self.albedo = np.clip(gaussian_filter(albedo, sigma=0.7), 0.05, 0.45)
        return self.dem, self.albedo

    def _fractal_brownian_noise(self, size, octaves=6, persistence=0.55, lacunarity=2.0):
        terrain = np.zeros((size, size), dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0
        for _ in range(octaves):
            grid_dim = max(4, int(8 * frequency))
            rand_grid = self.rng.randn(grid_dim, grid_dim).astype(np.float32)
            layer = cv2.resize(rand_grid, (size, size), interpolation=cv2.INTER_CUBIC)
            terrain += amplitude * layer
            amplitude *= persistence
            frequency *= lacunarity
        # Normalize to [-1, 1]
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-7) * 2.0 - 1.0
        return terrain

    def _add_crater(self, dem, albedo, cx, cy, radius):
        """Add a crater with raised rim, parabolic bowl, and ejecta blanket."""
        size = dem.shape[0]
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Pike's lunar morphometric relation: depth d ~ 0.2 * Diameter = 0.4 * Radius
        depth = 0.35 * radius
        rim_height = 0.12 * radius

        # Parabolic cavity inside rim
        inner_mask = dist <= radius
        cavity = -depth * (1.0 - (dist / radius)**2) + rim_height
        dem[inner_mask] += cavity[inner_mask]

        # Central peak for large complex craters (R > 35)
        if radius > 35:
            peak_r = radius * 0.22
            peak_mask = dist <= peak_r
            peak_height = depth * 0.45 * np.exp(-((dist / (peak_r * 0.5))**2))
            dem[peak_mask] += peak_height[peak_mask]

        # Raised rim and ejecta blanket outside rim: falloff as r^-3
        rim_outer = radius * 2.6
        outer_mask = (dist > radius) & (dist <= rim_outer)
        ejecta = rim_height * ((radius / np.maximum(dist, 1e-5))**3.0)
        dem[outer_mask] += ejecta[outer_mask]

        # Albedo rays and bright impact ejecta for fresh craters
        freshness = self.rng.rand()
        if freshness > 0.4:
            bright_factor = 0.18 * freshness
            albedo[inner_mask] += bright_factor * 0.5
            albedo[outer_mask] += bright_factor * ((radius / np.maximum(dist, 1e-5))**2.0)[outer_mask]

        return dem, albedo

    def compute_surface_normals(self):
        """Compute pixel-wise surface normal vectors (nx, ny, nz) from DEM gradients."""
        # Using Sobel filters with proper meter scaling
        scale = self.resolution * 8.0
        dz_dx = cv2.Sobel(self.dem, cv2.CV_32F, 1, 0, ksize=3) / scale
        dz_dy = cv2.Sobel(self.dem, cv2.CV_32F, 0, 1, ksize=3) / scale

        # Surface normal n = (-p, -q, 1) / sqrt(p^2 + q^2 + 1)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(self.dem, dtype=np.float32)
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        return nx / norm, ny / norm, nz / norm

    def compute_cast_shadows(self, sun_azimuth_deg, sun_elevation_deg):
        """Ray-traced horizon shadow calculation for solar incidence."""
        az_rad = np.radians(sun_azimuth_deg)
        el_rad = np.radians(sun_elevation_deg)
        tan_el = np.tan(el_rad)

        dx = np.cos(az_rad)
        dy = np.sin(az_rad)

        # Max ray length across terrain
        max_dist = int(np.ceil(self.size * 0.75))
        shadow_mask = np.ones_like(self.dem, dtype=np.float32)

        # Horizon marching
        step_size = 1.5
        num_steps = int(max_dist / step_size)

        h, w = self.dem.shape
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)

        current_dem = self.dem
        for i in range(1, min(num_steps, 80)):
            dist = i * step_size
            sx = np.clip(np.round(x_indices + dist * dx).astype(np.int32), 0, w - 1)
            sy = np.clip(np.round(y_indices + dist * dy).astype(np.int32), 0, h - 1)
            ray_elevation = current_dem[sy, sx]
            min_elevation_needed = current_dem + dist * tan_el * self.resolution
            # If blocking terrain is higher than ray, current pixel is in shadow
            is_blocked = ray_elevation > min_elevation_needed
            shadow_mask[is_blocked] = 0.0

        return shadow_mask

    def render_image(self, sun_azimuth_deg=None, sun_elevation_deg=None, photometric_model='lommel_seeliger', **kwargs):
        """
        Render optical lunar image given sun angle and photometric parameters.
        Returns:
            image (H, W) float32 in [0, 1]
            shadow_mask (H, W) float32
        """
        if sun_azimuth_deg is None:
            sun_azimuth_deg = kwargs.get('sun_azimuth', 45.0)
        if sun_elevation_deg is None:
            sun_elevation_deg = kwargs.get('sun_elevation', 30.0)
        nx, ny, nz = self.compute_surface_normals()

        # Sun vector s
        az_rad = np.radians(sun_azimuth_deg)
        el_rad = np.radians(sun_elevation_deg)
        sx = np.cos(el_rad) * np.cos(az_rad)
        sy = np.cos(el_rad) * np.sin(az_rad)
        sz = np.sin(el_rad)

        # Nadir viewing vector v = (0, 0, 1)
        # Cosines of local incidence and emission angles
        mu_0 = np.maximum(0.0, nx * sx + ny * sy + nz * sz)
        mu = np.maximum(0.0, nz)
        phase_angle = np.pi / 2.0 - el_rad

        if photometric_model == 'lommel_seeliger':
            # Lommel-Seeliger law: R = mu_0 / (mu + mu_0)
            denom = mu + mu_0 + 1e-6
            reflectance = mu_0 / denom
        elif photometric_model == 'lunar_lambert':
            # Lunar-Lambert combination
            # L(g) lunar limb weight parameter
            L_g = np.exp(-phase_angle / np.radians(30.0))
            rls = mu_0 / (mu + mu_0 + 1e-6)
            rlambert = mu_0
            reflectance = 2.0 * L_g * rls + (1.0 - L_g) * rlambert
        else:
            # Standard Lambertian
            reflectance = mu_0

        # Cast shadows
        shadow_mask = self.compute_cast_shadows(sun_azimuth_deg, sun_elevation_deg)

        # Lunar ambient / diffuse inter-reflection (lunar umbra is near dark ~0.02)
        ambient = 0.035
        direct_flux = shadow_mask * (1.0 - ambient) + ambient

        # Optical image = Albedo * Reflectance * DirectFlux
        radiance = self.albedo * reflectance * direct_flux

        # Add realistic sensor photon shot noise (Poisson/Gaussian approx)
        noise = self.rng.normal(0, 0.006, radiance.shape).astype(np.float32)
        radiance = np.clip(radiance + noise, 0.0, 1.0)

        # Stretch/normalize contrast to standard 8-bit dynamic range [0, 1]
        p_low, p_high = np.percentile(radiance, (0.5, 99.5))
        if p_high > p_low:
            radiance = np.clip((radiance - p_low) / (p_high - p_low), 0.0, 1.0)

        return radiance.astype(np.float32), shadow_mask

    def sample_observation(self, sun_azimuth, sun_elevation, scale=1.0, crop_box=None):
        """
        Sample an observation at given illumination and spatial scale.
        crop_box: (x0, y0, w, h) in DEM coordinates.
        Returns:
            image: (H_out, W_out)
            meta: dictionary with transformation coordinates and scale
        """
        full_img, shadow = self.render_image(sun_azimuth, sun_elevation)
        if crop_box is None:
            # Default center crop to avoid border shadow effects
            margin = int(self.size * 0.15)
            w = h = self.size - 2 * margin
            x0 = y0 = margin
        else:
            x0, y0, w, h = crop_box

        cropped = full_img[y0:y0+h, x0:x0+w]

        # Apply scale (downsample or upsample)
        if scale != 1.0:
            target_w = max(16, int(w / scale))
            target_h = max(16, int(h / scale))
            scaled_img = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA if scale > 1.0 else cv2.INTER_CUBIC)
        else:
            scaled_img = cropped.copy()

        meta = {
            'x0': x0, 'y0': y0, 'w': w, 'h': h,
            'scale': scale,
            'sun_azimuth': sun_azimuth,
            'sun_elevation': sun_elevation,
            'original_shape': (h, w),
            'observed_shape': scaled_img.shape
        }
        return scaled_img, meta
