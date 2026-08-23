# Methodology

## 1. Direct historical registration

For a historical image \(I_h\), retrieve only current-map references \(I_r\). The configured matcher provides:

\[
\mathcal M_{hr}=\{(u_h^k,u_r^k,c_k,\sigma_k)\}.
\]

A current reference observation provides:

\[
u_r^k\leftrightarrow X_j^{current}.
\]

After uniqueness, provenance and mask checks, the historical image obtains:

\[
\mathcal C_h=\{(u_h^k,X_{j(k)}^{current})\}.
\]

The implementation does not count the same `point3D_id` multiple times when several references observe it.

## 2. Weighted PnP and pose modes

Each current reference first produces an independent pose hypothesis. Hypotheses are clustered by rotation and camera-center separation. If two competitive clusters remain, the image is `AMBIGUOUS_MULTIMODAL` and is not allowed to update the map.

For the dominant reference cluster, the refined pose minimizes:

\[
\hat T=\arg\min_T\sum_k w_k\rho\left(\|u_h^k-\pi(K_hTX_k)\|^2\right).
\]

The default weight is based on match/association confidence. A localizer integration may further calibrate confidence and predicted uncertainty.

## 3. Pose information

For a left pose perturbation \(\xi=[\delta t,\delta\theta]\), the projection Jacobian for a camera-frame point \(p_c=(x,y,z)\) is:

\[
J_\pi=
\begin{bmatrix}
 f_x/z & 0 & -f_xx/z^2\\
 0 & f_y/z & -f_yy/z^2
\end{bmatrix},
\]

\[
J_{motion}=\begin{bmatrix}I&-[p_c]_\times\end{bmatrix}.
\]

The normalized Fisher information is:

\[
\Lambda=\sum_k w_kJ_k^\top\Sigma_k^{-1}J_k.
\]

Translation columns are scaled with a characteristic scene length before computing the condition number. This avoids comparing metres directly with radians. The physical pose covariance is recovered from the inverse normalized information matrix.

The system reports:

- minimum/every eigenvalue;
- normalized condition number;
- `logdet` information volume;
- marginal translation and rotation standard deviations;
- convex-hull and grid support.

High inlier count alone is not sufficient.

## 4. Historical change direction

The current map is the truth direction:

- `CURRENT ∩ HISTORICAL` → stable and potentially usable;
- `HISTORICAL − CURRENT` → historical-only, never production geometry;
- `CURRENT − HISTORICAL` → irrelevant to the purpose of old-view augmentation.

A registered historical image receives an H×W label map:

- `INVALID=0`
- `STABLE=1`
- `CHANGED=2`
- `UNCERTAIN=3`

Only `STABLE` pixels can participate in future historical-reference PnP. `UNCERTAIN` is rejected, not silently treated as stable.

The executable baseline combines pose-aligned photometric and structural evidence. For paper-level experiments, inject DINOv2 or another dense feature extractor and fuse evidence across multiple current views.

## 5. Viewpoint-gap bridge

A direct failure is not evidence of change. If historical frames are internally consistent and image quality is acceptable, construct an image graph. Edge confidence combines match count, geometric inliers, inlier ratio, spatial support and epipolar quality.

The first bridge mechanism propagates fixed current point IDs through historical correspondences. Every historical frame with sufficient current IDs is independently re-localized by PnP. Pure relative-pose multiplication is only an initializer.

When current IDs no longer propagate, build a candidate old-view submap. Its old-only points remain quarantined. For monocular sessions, estimate:

\[
X^{current}=sRX^{old}+t
\]

with robust Sim(3), followed by an optimization in which every current pose and point remains fixed.

A production bridge normally requires two spatially separated current anchors or two approximately edge-disjoint paths, plus rotation/translation/scale cycle consistency. A single long chain remains a candidate.

## 6. Reference utility

A historical reference is included only if it improves future current-query localization:

\[
U(r)=
\alpha\Delta C_{view}+
\beta\Delta P_{localizer}+
\gamma\Delta I_{pose}+
\delta S_{stable}
-\eta R_{redundancy}
-\mu C_{runtime}
-\nu C_{risk}.
\]

Route cells discretize position, height, yaw, pitch, direction and condition. Greedy K-cover selection first fills cells below their support target, then adds references by marginal utility per cost. Direct and bridged references carry different risk priors.

The decisive measurement is held-out current-query localization gain, not how many historical images can be registered.

## 7. Long-term stability

Two scores are maintained separately:

- `geometry_currentness`: whether the associated structure still exists now;
- `historical_view_utility`: whether this old viewpoint still improves current localization.

Age may decay currentness evidence, but does not automatically remove a valuable historical view. A single unmatched event has zero penalty by default. Repeated geometric conflicts, repeated change evidence, or contribution to a confident wrong pose lower both scores and transition the reference through `SUSPECT` and `RETIRED` states.
