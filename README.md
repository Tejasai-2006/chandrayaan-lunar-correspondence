# Scale and Illumination Invariant Representations of Lunar Imagery

### Deep Unsupervised Representation Learning for Chandrayaan Optical Terrain Correspondence

---

## 1. Executive Summary & Problem Formulation

In lunar remote sensing missions—such as ISRO's **Chandrayaan-1** (Terrain Mapping Camera, TMC), **Chandrayaan-2**, and **Chandrayaan-3** (Orbiter High Resolution Camera, OHRC)—optical imagery of the identical surface terrain exhibits drastic structural and radiometric variations across orbital passes:
1. **Dynamic Solar Illumination**: The Moon has no atmosphere, resulting in absence of diffuse sky glow. Surface radiance is governed by direct solar incidence and extreme cast shadows (umbra). When the solar azimuth rotates ($\Delta \phi$), slope illumination reverses (a bright crater rim becomes a dark pit). When solar elevation varies ($\theta_{\text{sun}} \in [15^\circ, 75^\circ]$), shadows elongate or disappear completely.
2. **Spatial Scale & Altitude Shifts**: Orbital altitude fluctuations, cross-sensor multi-scale fusion (e.g. TMC $5\,\text{m/pixel}$ vs OHRC $0.25\,\text{m/pixel}$), and landing descent zooming induce substantial scale disparities ($1.0\times$ to $4.0\times$).
3. **Failure of Classical Baselines**: Classical intensity-based matching (Normalized Cross-Correlation / ZNCC) strictly assumes photometric brightness constancy ($I_1(x) \approx I_2(W(x))$). Under shadow inversion, raw pixel correlation drops to zero or becomes negative, locking onto transient shadow boundaries rather than invariant terrain geomorphology.

This project designs, trains, benchmarks, and delivers an end-to-end unsupervised learning framework to produce **scale- and illumination-invariant representations** of lunar terrain, measuring:
- **Neighbourhood preservation under transformation** (Top-$k$ Jaccard overlap, Spearman rank correlation $\rho$, Trustworthiness).
- **Reconstruction error** (MSE, PSNR, SSIM across varying solar elevations).
- **Matching stability** (Mean Matching Accuracy MMA@$\tau$, Repeatability, AUC vs solar azimuth and scale).
- **Mandatory Baseline**: Raw intensity correlation.
- **Interactive Deliverable**: Full-featured **Correspondence Explorer with Matched Pairs** web application.

---

## 2. Theoretical Formulation & Physics

### 2.1 Lunar Photometric Reflectance & Ray-Casting
Lunar regolith exhibits strong backward scattering modeled by the **Lommel-Seeliger law** and **Lunar-Lambert formulation**:
$$R_{LS}(\mu, \mu_0) = \frac{\mu_0}{\mu + \mu_0}$$
where:
- $\mu_0 = \max(0, \mathbf{n} \cdot \mathbf{s})$ is the cosine of the local solar incidence angle.
- $\mu = \max(0, \mathbf{n} \cdot \mathbf{v})$ is the cosine of the emission/viewing angle ($\mu = n_z$ for nadir orbital imagery).
- $\mathbf{s} = (\cos\theta_{\text{sun}} \cos\phi, \cos\theta_{\text{sun}} \sin\phi, \sin\theta_{\text{sun}})$ is the solar illumination vector.

Direct flux includes ray-traced horizon occlusion $S(x, y) \in \{0, 1\}$ computed by marching along the solar azimuth ray across the Digital Elevation Model (DEM):
$$I(x, y) = A(x, y) \cdot R_{LS}(\mu, \mu_0) \cdot \left[S(x, y)(1 - \epsilon_{amb}) + \epsilon_{amb}\right]$$

### 2.2 Unsupervised Representation Learning (Lunar-InvNet)
To decouple intrinsic topographical morphology from extrinsic illumination and scale, **Lunar-InvNet** employs a self-supervised convolutional architecture with:
1. **Instance / Group Normalization**: Removes patch-level intensity offsets and contrast scaling without relying on batch statistics.
2. **Multi-Scale Convolutional Receptive Fields**: Dual $3 \times 3$ and $5 \times 5$ kernels capture both micro-roughness and macro crater rims.
3. **Contrastive InfoNCE Objective**:
   $$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{\exp(z_i^A \cdot z_i^B / \tau)}{\sum_{j=1}^{B} \exp(z_i^A \cdot z_j^B / \tau)} + \log \frac{\exp(z_i^B \cdot z_i^A / \tau)}{\sum_{j=1}^{B} \exp(z_i^B \cdot z_j^A / \tau)} \right]$$
   where positive pairs $(z_i^A, z_i^B)$ are embeddings of the *same physical lunar patch* rendered under randomized solar angles ($\Delta \phi \in [30^\circ, 180^\circ]$, $\Delta \theta \in [15^\circ, 50^\circ]$) and zoom scales ($s \in [1.0, 2.5]$), mapped to the unit hypersphere $\mathcal{S}^{127}$.

---

## 3. Quantitative Evaluation Suite

### 3.1 Neighbourhood Preservation
Measures how well the topological manifold of feature space survives extreme environmental transformation:
- **Top-$k$ Jaccard Overlap**:
  $$J_k = \frac{1}{N} \sum_{i=1}^{N} \frac{|N_k(z_i^{\text{orig}}) \cap N_k(z_i^{\text{transf}})|}{|N_k(z_i^{\text{orig}}) \cup N_k(z_i^{\text{transf}})|}$$
- **Spearman Rank Correlation ($\rho$)**: Measures monotonic rank preservation between pairwise distance matrices $D^{\text{orig}}$ and $D^{\text{transf}}$.
- **Trustworthiness**: Quantifies avoidance of false neighbor intrusions.

### 3.2 Reconstruction Error
A convolutional autoencoder (**Lunar-AE**) is trained with latent bottleneck $z \in \mathbb{R}^{64}$ to reconstruct the intrinsic terrain morphology:
- Evaluates MSE, PSNR, and SSIM when encoding images under low grazing sun ($15^\circ$) versus sub-solar illumination ($70^\circ$).

### 3.3 Matching Stability
Evaluates Mutual Nearest Neighbor (MNN) correspondence matching against exact sub-pixel ground truth homography:
$$\text{MMA}@\tau = \frac{1}{|\mathcal{M}|} \sum_{(i, j) \in \mathcal{M}} \mathbb{I}\left( \|\mathbf{p}_j - \mathcal{H}(\mathbf{p}_i)\|_2 \le \tau \right)$$

---

## 4. Project Structure

```
Unsupervised Learning Project/
├── lunar_engine.py             # Topographic DEM generator & Lommel-Seeliger photometric renderer
├── representations.py          # Raw intensity baseline, Classical gradient, Lunar-InvNet, Lunar-AE
├── evaluation.py               # Metric suite: Neighbourhood preservation, MMA@tau, reconstruction
├── train_and_experiment.py     # Training pipeline, multi-parameter sweeps, benchmark artifacts
├── app.py                      # Interactive Streamlit "Correspondence Explorer with Matched Pairs"
├── utils.py                    # Keypoint detection, homography warping, match visualization
├── models/                     # Saved neural model weights (lunar_invnet, lunar_autoencoder)
├── results/                    # Benchmark JSON summaries, CSV sweeps, and high-res plots
└── README.md                   # Comprehensive scientific and architectural documentation
```

---

## 5. Quick Start & Execution

### 5.1 Run the Full Benchmark Suite
```bash
python train_and_experiment.py
```
This script will:
1. Generate synthetic lunar crater scenes and self-supervised contrastive training pairs.
2. Train Lunar-InvNet and Lunar-AE (or load existing checkpoints).
3. Execute parameter sweeps across solar azimuth ($0^\circ$ to $180^\circ$), solar elevation, and scale ($1.0\times$ to $3.0\times$).
4. Compute neighbourhood preservation, reconstruction metrics, and matching stability.
5. Save artifacts to `results/` and print the summary scorecard.

### 5.2 Launch the Correspondence Explorer
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with:
- Dynamic solar azimuth ($0^\circ - 360^\circ$), elevation ($15^\circ - 75^\circ$), and scale controls.
- Visual correspondence vectors with color-coded inlier/outlier verification lines.
- Side-by-side head-to-head comparison: Raw Intensity Baseline vs Lunar-InvNet.
- Interactive patch inspector explaining the physical failure mechanism of intensity correlation.
