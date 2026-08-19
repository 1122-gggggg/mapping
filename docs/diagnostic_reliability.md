# Diagnostic reliability and calibration

This note defines which MapDoctor outputs are **geometric screening signals**, which
outputs are **empirically calibrated probabilities**, and which outputs may be used as
conservative deployment gates. The distinction matters: a high SfM health score is not
itself a measured probability that an online localizer will succeed.

## 1. Evidence layers

The diagnostic stack intentionally separates four evidence layers.

1. **Static reconstruction evidence**: landmark support, track length, reprojection
   residuals, image-plane coverage, parallax, and image quality.
2. **Graph evidence**: hard cuts (isolated images, articulation images, bridge edges),
   soft bottlenecks (spectral connectivity), and sensitivity to the covisibility support
   threshold.
3. **Query evidence**: held-out retrieval/PnP outcomes, inlier geometry, pose consensus,
   and failure attribution.
4. **Deployment evidence**: cross-fitted failure calibration, risk--coverage curves, and
   a separately held-out operating-point audit.

Do not collapse these layers into one unqualified number. Static map evidence can rank
where to inspect or recapture, while only held-out query evidence can measure the actual
localizer/camera/environment combination.

## 2. Covisibility graph: exact support without Python pair expansion

For registered image \(i\) and landmark \(k\), define the binary incidence matrix

\[
B_{ik}=\mathbf 1[\text{image } i \text{ observes landmark } k].
\]

The exact number of shared landmarks between images is

\[
W = BB^\top,\qquad w_{ij}=|\mathcal L_i\cap\mathcal L_j|.
\]

The diagonal is discarded. The previous implementation explicitly expanded every
landmark track into all image pairs, requiring Python work proportional to
\(\sum_k {L_k\choose 2}\), where \(L_k\) is track length. The sparse-incidence product
computes the same counts in optimized SciPy sparse kernels and avoids materializing each
track pair in Python. The report retains `estimated_pair_expansions` so large-map runtime
pressure remains observable.

### 2.1 Hard cuts and soft bottlenecks

Tarjan articulation/bridge analysis detects only exact single-node or single-edge cuts.
A map can still contain two dense subgraphs connected by several very weak edges, so no
exact bridge exists even though optimization/localization support is fragile.

For thresholded weighted adjacency \(W\) and degree matrix \(D\), MapDoctor now reports
the second eigenvalue of the weighted normalized Laplacian

\[
L_{\mathrm{norm}}=I-D^{-1/2}WD^{-1/2},\qquad
\lambda_2=\lambda_2(L_{\mathrm{norm}}).
\]

A connected component has \(\lambda_2>0\); a value close to zero indicates a soft
bottleneck. This is a **within-map ranking signal**, not a universal pass threshold.
Its scale depends on graph construction, support weighting, sequence sampling density,
and the minimum-shared-landmark threshold.

The report therefore also recomputes connectivity near \(0.5\times\), \(1\times\), and
\(2\times\) the requested support threshold. A sudden component split under a modest
threshold increase is stronger evidence of fragile overlap than a single thresholded
component count.

## 3. Failure-probability calibration

Let \(s_i\in[0,1]\) be an existing map/query risk score and \(y_i\in\{0,1\}\) indicate a
strict localization failure. A score is calibrated when

\[
\Pr(Y=1\mid \hat p=p)\approx p.
\]

`mapdoctor calibrate-risk` evaluates three monotone mappings.

### 3.1 Identity

\[
\hat p(s)=s.
\]

Identity is retained as a candidate because a calibration procedure must be allowed to
conclude that the input score is already better than a fitted alternative.

### 3.2 Binomial isotonic regression

Isotonic regression solves

\[
\min_{g}\sum_i (y_i-g(s_i))^2
\quad\text{subject to}\quad s_i\le s_j\Rightarrow g(s_i)\le g(s_j).
\]

The implementation uses the pool-adjacent-violators algorithm (PAVA). It is flexible and
preserves ranking, but can overfit small calibration sets.

### 3.3 Monotone beta calibration

The parametric map is

\[
\hat p(s)=\sigma\!\left(a\log s-b\log(1-s)+c\right),
\qquad a\ge0,\ b\ge0,
\]

where \(\sigma(z)=1/(1+e^{-z})\). The non-negative constraints preserve monotonic risk
ordering. The identity map is included at \(a=b=1,c=0\), unlike ordinary logistic/Platt
calibration. A small L2 penalty shrinks the fit toward identity.

### 3.4 Group cross-fitting

Adjacent video frames are strongly correlated. Random frame-level folds can place nearly
identical views in train and validation sets, producing an optimistic Brier score and an
overconfident failure probability.

Let \(G(q_i)\) be a session, flight, route segment, or pre-declared spatial block. The
cross-fitting rule is

\[
G(q_i)=G(q_j)\Rightarrow \operatorname{fold}(q_i)=\operatorname{fold}(q_j).
\]

For each fold, the calibrator is fit on all other groups and predicts the held-out groups.
`auto` selects the available method with the lowest out-of-fold Brier score, breaking
numerical ties in favor of the simpler model. The output distinguishes:

- `out_of_fold_risks`: use these to evaluate this calibration dataset;
- `final_calibrator`: fit on all calibration rows, use only for future untouched data;
- `calibrated_risks`: in-sample predictions from the final model, not an unbiased metric.

Preferred grouping order is independent flight/session IDs, then non-overlapping route
segments, then coarse spatial blocks. Query-level folds are allowed only as a diagnostic
fallback and emit a leakage warning.

## 4. Calibration metrics

For predicted failure probabilities \(p_i\), the report includes

\[
\operatorname{Brier}=\frac1N\sum_i(p_i-y_i)^2,
\]

\[
\operatorname{LogLoss}=-\frac1N\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right],
\]

and adaptive expected calibration error

\[
\operatorname{ECE}=\sum_b\frac{|B_b|}{N}
\left|\overline p_b-\overline y_b\right|.
\]

ECE depends on binning and must not be used alone. Brier score measures both calibration
and discrimination; log loss strongly penalizes confident mistakes. Inspect all three,
the reliability bins, AUROC, and risk--coverage behavior.

## 5. Selective localization and operating points

If low-risk queries are accepted first, selective risk at coverage \(c=k/N\) is

\[
R(k)=\frac{1}{k}\sum_{i=1}^{k} y_{(i)},
\]

where \((i)\) denotes ascending predicted risk. AURC averages \(R(k)\) over all prefixes.
Equal-score groups are handled without arbitrary query-name ordering; operational
thresholds never split a tie group.

`operating_points` are empirical and may be optimistic. `safe_operating_points` add a
one-sided Clopper--Pearson upper bound and Bonferroni correction over all complete score
thresholds. This is a conservative engineering audit, not an unconditional guarantee.
Strict validity requires all of the following:

1. the risk model and calibrator were fixed without using the certification labels;
2. the certification units are independent and identically distributed, or exchangeable
   under the deployment distribution;
3. score thresholds are label-independent;
4. camera, map, localizer, environment, and strict-failure definition match deployment.

For video, adjacent frames are not independent certification units. Use independent
flights/sessions or pre-declared non-overlapping blocks. For formal distribution-free
risk control under adaptive selection, use a conformal-risk-control method on a dedicated
calibration/certification split.

## 6. Recommended workflow

### 6.1 Map-only screen

```bash
sfm-qa analyze /path/to/sparse/0 --backend gluemap --output qa-out
mapdoctor graph-fragility /path/to/sparse/0 --backend gluemap \
  --minimum-shared-landmarks 15 --output graph.json
```

Use graph cuts, spectral connectivity, threshold sensitivity, parallax, FIM directions,
coverage, and weak-region causes to decide where to inspect or recapture. Do not enable a
new absolute gate until its threshold has been validated on held-out localization runs.

### 6.2 Build leakage-resistant groups

Prefer an explicit JSON mapping:

```json
{
  "flight01/frame0001.jpg": "flight01",
  "flight01/frame0002.jpg": "flight01",
  "flight02/frame0001.jpg": "flight02"
}
```

Then cross-fit calibration:

```bash
mapdoctor calibrate-risk loc.csv raw_risk.json \
  --groups session_groups.json --folds 5 --min-samples 20 \
  --output calibration.json \
  --scores-output calibrated_oof_risk.json
```

If session IDs are unavailable and every query has metric `x,y,z`, a spatial fallback is
available:

```bash
mapdoctor calibrate-risk loc.csv raw_risk.json \
  --spatial-block-size 10 --folds 5
```

### 6.3 Evaluate ranking, calibration, and conservative coverage

```bash
mapdoctor risk-coverage loc.csv calibrated_oof_risk.json \
  --ece-binning equal_mass --confidence 0.95 \
  --target-failure-rate 0.01 0.02 0.05 \
  --output risk_coverage.json
```

This evaluates out-of-fold predictions. The serialized final calibrator can be applied to
future raw scores without reusing labels:

```bash
mapdoctor apply-risk-calibrator calibration.json future_raw_risk.json \
  --output future_calibrated_risk.json
```

A final deployment threshold still requires a separate untouched certification set; do
not select and certify the threshold on the same labels.

## 7. What this PR deliberately does not claim

- A low reprojection error does not imply strong localization geometry.
- A full-rank bearing FIM is not a calibrated pose-error covariance when landmark/map
  uncertainty and correlated observations are ignored.
- A high readiness score is not a success probability.
- A positive \(\lambda_2\) only proves graph connectivity; it does not prove that image
  matching, triangulation, or PnP will succeed.
- Cross-fitted frame predictions do not make adjacent frames independent for final safety
  certification.
- The system diagnoses an existing map/localizer. It does not replace held-out flights,
  route replay, or ground-truth pose evaluation.

## 8. Literature basis and stronger extensions

The implementation is intentionally narrower than several stronger research methods:

- **Map Quality Evaluation for Visual Localization** (ICRA 2017) motivates evaluating a
  map by downstream localization behavior rather than reconstruction appearance alone.
- **Fisher Information Field** and later information-aware view planning use pose
  observability over space; MapDoctor's FIM is a lightweight local proxy, not a learned
  field or full uncertainty-aware estimator.
- **ActLoc** learns viewpoint-conditioned localization accuracy and optimizes camera
  orientation along a trajectory. With sufficient representative labels, it can model
  appearance and viewpoint effects that hand-designed FIM/coverage metrics miss.
- **Spectral Measurement Sparsification for Pose-Graph SLAM** connects algebraic
  connectivity to estimation quality. MapDoctor uses \(\lambda_2\) only as a diagnostic;
  it does not solve the edge-selection optimization.
- **Beta calibration** motivates a monotone parametric family that contains identity.
- Spatial/temporal blocking literature motivates group-level validation whenever samples
  are autocorrelated.
- Selective conformal risk-control methods provide stronger finite-sample guarantees than
  an empirical AURC or a binomial confidence bound, but require a carefully separated
  calibration/certification protocol.

Primary references:

1. Merzić et al., *Map Quality Evaluation for Visual Localization*, ICRA 2017.
2. Zhang et al., *Fisher Information Field: An Efficient and Differentiable Map for
   Perception-aware Planning*, 2020/2021.
3. Li et al., *ActLoc: Learning to Localize on the Move via Active Viewpoint Selection*,
   CoRL 2025.
4. Doherty, Rosen, and Leonard, *Spectral Measurement Sparsification for Pose-Graph
   SLAM*, 2022.
5. Kull, Silva Filho, and Flach, *Beta Calibration*, AISTATS 2017.
6. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
7. Roberts et al., *Cross-validation strategies for data with temporal, spatial,
   hierarchical, or phylogenetic structure*, Ecography 2017.
8. Angelopoulos et al., *Conformal Risk Control*, 2022/2024.
