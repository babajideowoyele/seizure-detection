# Video-Based Pediatric Seizure Detection from Pose Landmark Sequences

**Project summary — 3-page brief**

Babajide Owoyele, Sharjeel Shaikh, Wim Pouw, Moritz Schneider, Stefan Konigorski, Gerard de Melo

*Hasso Plattner Institute, Universität Potsdam, Germany*

---

## Abstract

We address binary detection of *infantile spasms* — the defining seizure type of West syndrome — from short, pose-only video segments of pediatric subjects. Each input is a $150 \times 33 \times 5$ tensor of MediaPipe Pose landmarks (≈5 s at 30 fps; 33 body landmarks; $x,y,z$ plus visibility and presence). The pose-only representation preserves patient identity while retaining the kinematic signal clinicians use to recognise a spasm. Our pipeline combines landmark preprocessing, 518-dim engineered features, and a 5-fold ensemble of two complementary temporal architectures (CNN-GRU and dilated TCN) trained with focal loss and child-grouped cross-validation. On 5-fold `StratifiedGroupKFold`, mean validation $F_1$ is **0.628** (CNN-GRU) and **0.635** (TCN). On the held-out challenge test set our entry attained $F_1 = 0.2975$, placing 19th of 26 valid submissions (top score 0.4502). The work was submitted to the 2026 *Video-Based Seizure Detection Challenge* organised by the Section on Computational Neurology at Charité — Universitätsmedizin Berlin in collaboration with the International Conference on AI in Epilepsy and Other Neurological Disorders.

## 1. Problem

*Infantile spasms* are the defining seizure type of West syndrome — an age-dependent epileptic encephalopathy with typical onset between 3 and 24 months. The dominant motor pattern is a brief (1–3 s) sudden contraction of the trunk and limbs (flexor, extensor, or mixed), often occurring in clusters. Time-to-treatment strongly predicts cognitive outcome, and home-video review is part of the standard diagnostic pathway because the spasms are short, intermittent, and most often first observed by caregivers. Two constraints shape any deployable system: patient identifiability rules out raw-pixel transmission, and recordings are typically long. A pose-only intermediate representation collapses each frame to a few hundred numbers, removes pixel-level identity, and is approximately invariant to lighting, clothing, and background. The 2026 *Video-Based Seizure Detection Challenge* distributes exactly this representation: per-segment NumPy arrays of anonymised MediaPipe Pose landmarks.

## 2. Approach

<figure>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 690" style="max-width:56%;height:auto;display:block;margin:0.3em auto">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L0,8 L6,4 z" fill="#222"/>
    </marker>
  </defs>
  <style>
    .b   { fill:#fff; stroke:#333; stroke-width:1.1; }
    .ti  { font:italic 13px Charter,Cambria,Palatino,Georgia,serif; fill:#000; }
    .sub { font: 10.5px Charter,Cambria,Palatino,Georgia,serif; fill:#555; }
    .l   { stroke:#333; stroke-width:1.1; fill:none; }
  </style>
  <rect class="b" x="40"  y="20"  width="190" height="60" rx="2"/>
  <text class="ti"  x="135" y="44" text-anchor="middle">.npy (150,33,5)</text>
  <text class="sub" x="135" y="62" text-anchor="middle">challenge input</text>
  <text class="ti"  x="250" y="56" text-anchor="middle">or</text>
  <rect class="b" x="270" y="20"  width="190" height="60" rx="2"/>
  <text class="ti"  x="365" y="44" text-anchor="middle">raw video</text>
  <text class="sub" x="365" y="62" text-anchor="middle">MediaPipe adapter</text>
  <path class="l" d="M135,80 L135,100 L250,100 L250,120" marker-end="url(#arr)"/>
  <path class="l" d="M365,80 L365,100 L250,100"/>
  <rect class="b" x="130" y="120" width="240" height="64" rx="2"/>
  <text class="ti"  x="250" y="142" text-anchor="middle">Preprocess</text>
  <text class="sub" x="250" y="160" text-anchor="middle">NaN forward/back-fill</text>
  <text class="sub" x="250" y="176" text-anchor="middle">hip-center · torso-normalize</text>
  <line class="l" x1="250" y1="184" x2="250" y2="212" marker-end="url(#arr)"/>
  <rect class="b" x="100" y="212" width="300" height="78" rx="2"/>
  <text class="ti"  x="250" y="234" text-anchor="middle">Features  150 × 518</text>
  <text class="sub" x="250" y="252" text-anchor="middle">raw · velocity · acceleration · speed</text>
  <text class="sub" x="250" y="266" text-anchor="middle">joint angles · symmetry · NaN flag</text>
  <text class="sub" x="250" y="280" text-anchor="middle">low-frequency FFT power</text>
  <path class="l" d="M250,290 L250,312 L130,312 L130,332" marker-end="url(#arr)"/>
  <path class="l" d="M250,290 L250,312 L370,312 L370,332" marker-end="url(#arr)"/>
  <rect class="b" x="35"  y="332" width="190" height="76" rx="2"/>
  <text class="ti"  x="130" y="354" text-anchor="middle">CNN–GRU × 5 folds</text>
  <text class="sub" x="130" y="372" text-anchor="middle">1D-CNN → BiGRU(2)</text>
  <text class="sub" x="130" y="386" text-anchor="middle">→ attention → MLP</text>
  <rect class="b" x="275" y="332" width="190" height="76" rx="2"/>
  <text class="ti"  x="370" y="354" text-anchor="middle">TCN × 5 folds</text>
  <text class="sub" x="370" y="372" text-anchor="middle">5 dilated blocks (1→16)</text>
  <text class="sub" x="370" y="386" text-anchor="middle">→ attention → MLP</text>
  <path class="l" d="M130,408 L130,430 L250,430 L250,452" marker-end="url(#arr)"/>
  <path class="l" d="M370,408 L370,430 L250,430"/>
  <rect class="b" x="130" y="452" width="240" height="56" rx="2"/>
  <text class="ti"  x="250" y="474" text-anchor="middle">TTA  × 7 variants</text>
  <text class="sub" x="250" y="492" text-anchor="middle">reverse · jitter · scale ±5% · shift ±3</text>
  <line class="l" x1="250" y1="508" x2="250" y2="536" marker-end="url(#arr)"/>
  <rect class="b" x="130" y="536" width="240" height="50" rx="2"/>
  <text class="ti"  x="250" y="558" text-anchor="middle">Ensemble mean</text>
  <text class="sub" x="250" y="574" text-anchor="middle">average of 10 × 7 = 70 probabilities</text>
  <line class="l" x1="250" y1="586" x2="250" y2="614" marker-end="url(#arr)"/>
  <rect class="b" x="130" y="614" width="240" height="50" rx="2"/>
  <text class="ti"  x="250" y="636" text-anchor="middle">Threshold  τ = 0.65</text>
  <text class="sub" x="250" y="652" text-anchor="middle">label ∈ {0, 1}</text>
</svg>
<figcaption style="font-style:italic;color:#555;text-align:center;font-size:0.88em;margin-top:0.2em;">Figure 1. End-to-end pipeline.</figcaption>
</figure>

**Preprocessing.** Each sample is hip-centered (mid-hip = mean of MediaPipe landmarks 23 and 24), scaled by torso length (mid-shoulder to mid-hip), and `NaN` frames are forward/back-filled. A binary "frame-was-missing" flag is preserved as a feature.

**Features.** A 518-dim per-frame vector concatenates raw landmarks (165), velocity and acceleration (99 each), per-landmark speed (33), 10 joint angles, 6 left–right symmetry distances, the `NaN` mask, and 105 low-frequency FFT power components for seven kinematic landmarks broadcast across all 150 frames. Per-feature mean/std are computed once on the training set and persisted.

**Models.** Two architectures share an attention-pooling head and a small MLP classifier. The **CNN-GRU** is a 1D-CNN front-end (kernels 5 and 3) feeding a 2-layer bidirectional GRU. The **TCN** is five residual blocks with dilations $\{1, 2, 4, 8, 16\}$ giving full coverage of the 150-frame window. They have different inductive biases (recurrent vs.\ parallel multi-scale) and their per-fold thresholds occupy different points on the precision-recall curve, motivating an ensemble.

**Training.** `StratifiedGroupKFold` ($k=5$) grouped by patient ID prevents within-subject leakage. Focal loss ($\alpha=0.75$, $\gamma=2.0$) over a class-balanced `WeightedRandomSampler`. AdamW + OneCycleLR. Up to 120 epochs with early stopping on validation $F_1$ (patience 25). Augmentation: time reversal, Gaussian noise, frame dropout, time-index resampling, and same-class mixup.

**Inference.** Seven test-time-augmentation variants (original, reverse, jitter, ±5% scale, ±3-frame shift). Ten ensemble members × seven variants = 70 forward passes per sample, mean-pooled. Threshold $\tau=0.65$ tuned on cross-validation. A video adapter (`video_to_features.py`) renders raw `.mp4` input to the $(150, 33, 5)$ form the trained models expect, so the same checkpoint runs unmodified on clinical recordings.

## 3. Results

**Cross-validation (training partition).** 5-fold `StratifiedGroupKFold` grouped by patient ID:

| Fold | CNN-GRU $F_1$ | CNN-GRU $\hat\tau$ | TCN $F_1$ | TCN $\hat\tau$ |
|-----:|------:|------:|------:|------:|
|    0 | 0.633 | 0.70  | 0.611 | 0.63  |
|    1 | 0.615 | 0.59  | 0.608 | 0.61  |
|    2 | 0.629 | 0.64  | 0.635 | 0.33  |
|    3 | 0.575 | 0.28  | 0.627 | 0.25  |
|    4 | 0.691 | 0.51  | 0.694 | 0.39  |
| **Mean** | **0.628** | 0.54 | **0.635** | 0.44 |

Mean CV $F_1$ is comparable between architectures (0.628 vs 0.635), but per-fold thresholds differ: CNN-GRU prefers higher-precision operating points (0.51–0.70 on 4 of 5 folds); TCN prefers lower thresholds (0.25–0.39 on 3 of 5 folds). This decorrelation is the source of the ensemble's gain.

**Held-out test set.** Our submission attained $F_1 = 0.2975$, placing 19th of 26 valid entries. The top three by $F_1$ were Kramer (BU/JHU, 0.4502, two-step pipeline), Ramesh (NIT Tiruchirappalli, 0.4286, 1D-CNN+LightGBM), and Daraie (JHU, 0.4048, 1D-ResNet+XGBoost). No submission exceeded $F_1 = 0.5$. The gap from our CV $F_1 \approx 0.63$ to held-out $0.2975$ is consistent with (i) distribution shift between training- and test-pool subjects, (ii) threshold over-fitting to CV folds, (iii) the well-known sensitivity of $F_1$ to small precision/recall shifts on imbalanced binary tasks, and (iv) sub-second concentration of spasm motor energy that full-window attention pooling may under-exploit relative to multiple-instance-learning architectures used by some top entries.

## 4. Compute and Implementation

The pipeline is pure PyTorch with NumPy/Pandas/scikit-learn. Training is one-shot — a single full pass produces all 10 ensemble checkpoints, the feature normalization statistics, and a JSON results file with per-fold thresholds. The system is delivered as a Docker image (`Dockerfile` in the repo; pushed to GHCR via a manual-dispatch GitHub Actions workflow) and runs unchanged on CPU for inference, though training requires a single CUDA GPU. Total training run, including the 5-fold × 2-architecture grid and the per-epoch validation pass, is bounded by the 120-epoch budget × patience-25 early stopping; in practice early stopping fires well before 120 epochs on most folds.

The video-adapter path adds `mediapipe` and `opencv-python-headless` as inference-time dependencies and is decoupled from the competition Docker entry point so the submitted image stays minimal.

## 5. Discussion and Future Directions

**Pose-only ceiling.** Operating purely in pose space buys privacy, lighting invariance and lightness at the cost of losing facial cues, vital signs and contextual reactions. $F_1 \approx 0.63$ on 5-fold CV is consistent with a moderate-strength pose-only system on a small, imbalanced dataset.

**Three natural next steps.** (i) Replace MediaPipe with the metric-scale, 87-joint MeTRAbs estimator: richer joints expose hand and head dynamics that the 33-landmark MediaPipe topology loses, and metric scale removes the need for torso normalisation. (ii) Adopt a multi-view triangulation pipeline based on recent work in markerless dynamic extrinsic calibration [1] to fuse synchronised camera views and eliminate single-camera occlusion failures. (iii) Compose the detector with the *MaskAnyone* toolkit [2] developed in our group so that home-video material submitted by caregivers can be processed and classified entirely on-device, with only landmark coordinates and the binary detection label leaving the patient's premises.

**Acknowledgement.** This work was supported by the Federal Ministry of Research, Technology and Space under the funding code "KI-Servicezentrum Berlin-Brandenburg" (16IS22092). Responsibility for the content of this publication remains with the authors. We thank the Section on Computational Neurology at Charité — Universitätsmedizin Berlin and the organisers of the 2026 International Conference on AI in Epilepsy and Other Neurological Disorders for hosting the challenge.

## References

[1] de la Place, F. *Dynamic Extrinsic Camera Calibrator.* https://github.com/flodelaplace/lab-camera-dynamic-calibrator.

[2] Owoyele, B. A., Schilling, M., Sawahn, R., Kaemer, N., Zherebenkov, P., Verma, B., Pouw, W., & de Melo, G. (2024). *MaskAnyone toolkit: offering strategies for minimizing privacy risks and maximizing utility in audio-visual data archiving.* Proceedings of the 45th International Conference on Information Systems (ICIS), Bangkok. https://aisel.aisnet.org/icis2024/adv_theory/adv_theory/1/.

[3] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV. doi:10.1109/ICCV.2017.324.

[4] Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.* arXiv:1803.01271.
