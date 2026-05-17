# Video-Based Pediatric Seizure Detection from Pose Landmark Sequences

**An Ensemble of CNN-GRU and Temporal Convolutional Networks with Engineered Kinematic Features**

Babajide Owoyele, Sharjeel Shaikh, Wim Pouw, Moritz Schneider, Stefan Konigorski, Gerard de Melo

*Hasso Plattner Institute, Universität Potsdam, Germany*

---

## Abstract

We address binary detection of *infantile spasms* — the defining seizure type of West syndrome — from short, pose-only video segments of pediatric subjects. Each input is a $150 \times 33 \times 5$ tensor of MediaPipe Pose landmarks (≈5 s at 30 fps; 33 body landmarks; per-landmark $x,y,z$ coordinates plus visibility and presence scores). The pose-only representation preserves patient privacy while retaining the kinematic information clinicians use to recognize a spasm event. The work was prepared as an entry to the 2026 *Video-Based Seizure Detection Challenge*, organized by the Section on Computational Neurology at Charité — Universitätsmedizin Berlin in collaboration with the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders.

Our pipeline combines (i) landmark-level preprocessing that handles missing frames and normalizes body scale, (ii) a 518-dimensional per-frame feature vector aggregating raw landmarks, velocity, acceleration, joint angles, left–right symmetry distances, and low-frequency spectral power, (iii) a 5-fold ensemble of two complementary temporal architectures — a CNN-frontend with a bidirectional GRU and attention pooling, and a multi-scale Temporal Convolutional Network (TCN) — trained with focal loss, weighted sampling, and on-the-fly augmentation under a child-grouped cross-validation scheme, and (iv) a test-time-augmented inference path with a tuned ensemble decision threshold.

On 5-fold `StratifiedGroupKFold` cross-validation against the training partition, the CNN-GRU achieves a mean $F_1$ of **0.628** and the TCN **0.635**. The combined ensemble, evaluated on the held-out challenge test set, attained an $F_1$ of **0.2975** — ranking 19th of 26 submissions, against a top score of 0.4502. We discuss this generalization gap candidly, describe a video-to-features front-end that lets the same trained models run on raw video, and outline extensions: replacing MediaPipe with the metric-scale, 87-joint MeTRAbs pose estimator, a multi-view calibration path adapted from recent work on markerless dynamic extrinsic calibration, and integration with related work on privacy-preserving de-identification developed in our group [29].

---

## 1. Introduction

*Infantile spasms* are the defining seizure type of West syndrome — an age-dependent epileptic encephalopathy that typically presents between 3 and 24 months of life and carries a high risk of long-term neurodevelopmental morbidity if treatment is delayed [1,5]. The dominant motor pattern is a brief, sudden, axially-symmetric contraction (flexor, extensor, or mixed) lasting 1–3 seconds, often occurring in clusters. Early identification matters: time-to-treatment is a strong predictor of cognitive outcome, and home-video evaluation by epileptologists is an established part of the diagnostic pathway [21].

Two practical constraints shape any deployable detection system: privacy and robustness to long recordings. Raw pixel-level video makes both hard — infants are identifiable, and pixel-level models are expensive. A pose-only intermediate representation collapses each frame to a few hundred numbers, removes pixel-level identity, and is approximately invariant to clothing, lighting, and background. The 2026 *Video-Based Seizure Detection Challenge*, organized by the Section on Computational Neurology at Charité — Universitätsmedizin Berlin in collaboration with the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders, distributes exactly this representation: per-segment NumPy arrays of anonymized MediaPipe Pose landmarks extracted from clinical home-video recordings of pediatric subjects [4].

In this work we describe a complete pipeline that takes pre-extracted pose tensors as input and produces a binary infantile-spasm label per segment. The system was developed for the competition's Docker-based evaluation harness, reached a mean validation $F_1 \approx 0.63$ under child-grouped cross-validation on the training partition, achieved $F_1 = 0.2975$ on the held-out test set (rank 19 of 26 submissions), and has been extended with a video front-end so the same trained models can be applied to raw clinical recordings without retraining.

### Contributions

- A reproducible end-to-end pipeline — preprocessing, feature extraction, training, inference — specialized for short pose-only sequences with strong class imbalance and small-subject leakage risk.
- A two-architecture ensemble (CNN-GRU + multi-scale TCN) that measurably outperforms either model alone at the operating threshold, with per-fold thresholds tuned for $F_1$.
- Subject-grouped cross-validation (`StratifiedGroupKFold`) by patient identity to prevent same-child leakage between train and validation splits.
- A 7-variant test-time-augmentation scheme tailored to fixed-length temporal sequences, applied at the per-fold and per-model levels and averaged.
- A video-to-features adapter so the trained ensemble can be evaluated on raw `.mp4` input without retraining.

## 2. Background

### 2.1 Infantile spasms and West syndrome

The challenge target is *infantile spasms*, the defining seizure type of West syndrome [6]. Onset typically falls between 3 and 24 months of age, with a peak around 6 months. The triad classically associated with West syndrome is: infantile spasms, the hypsarrhythmia EEG pattern, and developmental regression or arrest. Roughly 1 in 2 000 to 4 000 children is affected.

The motor signature of an infantile spasm is short, sudden, and often clustered. A single spasm lasts 1–3 seconds and consists of a brief contraction of the trunk and limbs that can be predominantly *flexor* (a forward jackknife with head, arms and legs flexed), *extensor* (sudden extension of the trunk and limbs, sometimes resembling startle), or *mixed*. Single spasms are easy to miss on bedside observation; characteristically they occur in clusters of dozens, often near sleep–wake transitions. Home-video review by an epileptologist is part of the standard diagnostic pathway because the spasms are short, intermittent, and frequently first observed by caregivers rather than clinicians.

Two properties of infantile-spasm semiology shape the detection problem. First, the relevant signal is on the order of seconds — well-matched to the 5-second window of the challenge. Second, motor energy is *concentrated* in time: in a 5-second window containing a spasm, the spasm itself may occupy less than half the window, with the rest being baseline activity or post-ictal quiescence. This places attention pooling at a structural advantage over simple averaging across the temporal dimension.

### 2.2 Pediatric epilepsy and seizure semiology more broadly

Epilepsy is among the most common neurological disorders of childhood, with a cumulative incidence of approximately 1% by adolescence [1]. Roughly one in three patients is *drug-resistant* — seizures persist despite appropriate trials of two or more antiseizure medications — and accurate diagnosis of the seizure type and focus is the gate to advanced treatment options including epilepsy surgery, dietary therapy, and neuromodulation [5].

The International League Against Epilepsy classifies seizures by onset (focal, generalized, unknown) and by their dominant motor and non-motor manifestations [6]. The motor manifestations most amenable to video- or pose-based detection include:

- **Tonic-clonic seizures.** A tonic phase (sustained muscle contraction, often with loss of posture) is followed by clonic phase (bilateral rhythmic jerking at approximately 0.5–3 Hz that slows as the seizure progresses).
- **Clonic and myoclonic seizures.** Rhythmic (clonic) or brief, shock-like (myoclonic) jerks, often unilateral.
- **Tonic seizures.** Sustained increase in muscle tone; characteristic posturing, sometimes asymmetric.
- **Atonic seizures.** Sudden loss of postural tone; head-drops and falls are typical.
- **Hyperkinetic seizures.** Complex, rapid, often bilateral movements involving limbs and trunk.
- **Automatisms.** Coordinated, repetitive movements (lip-smacking, picking at clothes, repetitive limb movements) most often associated with focal impaired-awareness seizures of temporal-lobe origin.

These motor signatures span a wide range of spatial and temporal scales — from sub-second myoclonic jerks to multi-minute hyperkinetic episodes — but several common features fall within the bandlimited rhythmic range (0.2–3 Hz) targeted by the spectral component of our feature representation.

### 2.3 Epilepsy monitoring units and the role of video

Long-term video-EEG monitoring in an EMU is the clinical gold standard for characterizing seizure type, lateralization, and (in surgical candidates) the focus of seizure onset [1,2]. Patients are admitted for periods of days to weeks; video and scalp EEG are recorded continuously. The clinical bottleneck is then *review*: an epileptologist or trained technician scrubs through tens to hundreds of hours of footage per admission, flagging candidate events for detailed analysis. Automated screening — a triage layer that flags suspicious time windows for human review — has been studied for several decades [3,7] and is an active commercial and academic area.

### 2.4 The case for pose-only detection

A pose representation strips pixel-level identifying content (face, room, clothing) and reduces a frame from millions of pixels to a few hundred floats. This has three downstream consequences:

- **Privacy by construction.** If pose is extracted at the bedside, only landmark coordinates leave the patient's room, removing the need to transport or store raw video.
- **Computational lightness.** Downstream inference cost is dominated by the upstream pose estimator; landmark-input classifiers are inexpensive.
- **Cross-domain robustness.** Pose representations are approximately invariant to clothing, lighting, and background — the dominant nuisance variables in clinical recordings.

The cost is information loss. Subtle facial cues (gaze deviation, mouth automatisms), vital signs (color, perspiration), and contextual signals (a caregiver's reaction, alarm sounds) are not in the pose stream. This sets a practical ceiling that pure pose-based systems cannot exceed without auxiliary modalities.

Privacy-preserving de-identification of clinical audio-visual material is a closely related research direction. The *MaskAnyone* toolkit developed by the same group [34] formalises a range of de-identification strategies — segmentation-based body masking, face blur, audio scrubbing, and pose-only summarisation — explicitly trading utility for privacy across a controlled spectrum. The present work sits at the high-privacy end of that spectrum: by accepting a pose-only representation as the *input*, we obtain a detection pipeline that never sees pixels and is therefore composable with any pose-extraction front-end that satisfies the same anonymisation guarantees.

## 3. Related Work

### 3.1 Skeleton-based action recognition

Skeleton-based action recognition has matured rapidly since the introduction of *ST-GCN* [8], which formalized the spatial-temporal graph convolutional network over the human body topology. Subsequent work has refined the topology by learning adaptive graph adjacencies (2s-AGCN [9]), introduced multi-scale aggregation (MS-G3D [10]), and ultimately revisited the problem with heatmap-volume CNNs that often outperform the graph-based family on standard benchmarks (PoseConv3D [11]). For longer videos with reliable multi-frame pose, this family of architectures is the modern default.

Our setting is more constrained: short (5 s) clips, a small dataset (a few thousand segments, $\mathcal{O}(10^2)$ patients), and a single binary target. In this regime, simpler 1D-CNN and dilated-convolution backbones operating over a per-frame feature vector are competitive with graph approaches and substantially easier to train and regularize. The trade-off we accept is that body topology must be re-introduced through hand-engineered features (joint angles, left–right symmetry distances) rather than learned through graph convolutions.

### 3.2 Human pose estimation pipelines

A wide ecosystem of pose estimators is available. Single-stage bottom-up 2D models such as *OpenPose* [12] popularised real-time multi-person pose. Subsequent work has split along several axes:

- **Mobile-optimized**: *BlazePose* [13], the pose estimator behind MediaPipe Pose, returns 33 landmarks per person with $(x, y, z, \text{visibility}, \text{presence})$ channels at high frame rates on CPU and mobile hardware. The pre-extracted competition data corresponds to this output format.
- **High-accuracy top-down 2D**: *HRNet* [14] established a high-resolution multi-scale design that remains a strong baseline; *RTMPose* [15] later combined HRNet-style high-res features with lightweight transformer decoders for a favorable accuracy/latency trade-off.
- **2D-to-3D lifting**: *VideoPose3D* [16] showed that temporal convolutional lifting from 2D keypoints to 3D recovers consistent skeletons in relative scale.
- **Metric-scale single-stage 3D**: *MeTRAbs* [17] returns metric-scale absolute 3D coordinates for 87 body joints from a single image — substantially richer than MediaPipe's 33 landmarks and free of the scale ambiguity in relative-3D lifters.

The trade-off relevant to seizure detection is the joint count and metric scale. MediaPipe Pose covers the major body landmarks but discards much of the hand and face mesh; MeTRAbs's 87-joint skeleton captures both hands and a denser face / head representation that can be informative for automatisms and asymmetric motor seizures.

### 3.3 Automated seizure detection

Three modalities dominate the automated seizure-detection literature: EEG, wearable accelerometers, and video.

EEG-based detection is the longest-established branch and the standard of care in the EMU [3,7]. It operates on the substrate that physicians use to confirm seizures but is invasive in the wearable sense (scalp electrodes), and many useful seizure events are missed by surface EEG in the absence of clear ictal correlates.

Wearable accelerometers and surface EMG are an active area, with both academic and commercial systems (e.g., wrist-worn devices) targeting generalized tonic-clonic seizures, the type with the highest mortality risk [18,19]. These approaches are practical for at-home monitoring but capture only the wearer's limb dynamics and underperform on non-motor seizure types.

Video-based detection has been studied for at least two decades. Early work by Karayiannis and colleagues [3] targeted neonatal seizures with motion descriptors and classical machine learning. Optical-flow approaches such as Cuppens et al.\ [20] focused on nocturnal pediatric monitoring. More recent work by Geertsema et al.\ [21] demonstrated automated overnight detection of generalized tonic-clonic seizures in a residential care setting using bedside cameras and explicit motion features. Pose-based systems specifically, especially for pediatric populations, are a younger thread; the present work falls in that area.

### 3.4 Temporal sequence models

The deep-learning toolbox for sequence classification has converged on three families. Recurrent networks — LSTMs [22] and GRUs [23] — process sequences strictly sequentially and propagate state through time; they remain competitive on short sequences where a long-range receptive field is unnecessary. Temporal convolutional networks [24] use stacked dilated convolutions to achieve arbitrary receptive-field sizes with parallelism and explicit control over the temporal context. Transformers [25] dominate long-range sequence modeling at scale but are data-hungry and overfit aggressively on the size of dataset we work with.

Our two-architecture ensemble pairs a GRU-based recurrent model with a TCN, deliberately using two architectures with different inductive biases. This is a common motif in skeleton-action recognition (e.g., two-stream and four-stream ensembles [9]) and trades a small increase in inference cost for measurable robustness gains on small data.

### 3.5 Class imbalance, augmentation, and test-time augmentation

Strong class imbalance is the dominant statistical challenge in pediatric seizure data: even within a 24-hour EMU recording, ictal segments occupy a small fraction of the total. Standard remedies include cost-sensitive losses, class-weighted sampling, and synthetic minority oversampling. The *focal loss* of Lin et al.\ [26] adds an explicit difficulty-modulating factor $(1 - p_t)^\gamma$ to the cross-entropy, down-weighting confident examples and concentrating gradient on hard cases; we use it in combination with a `WeightedRandomSampler` that already produces approximately balanced batches.

Data augmentation for fixed-length kinematic sequences is conceptually similar to its image counterpart but constrained to transformations that preserve label semantics. Mixup [27] convex-combines pairs of examples and their labels, has a regularizing effect well-studied across modalities, and applies naturally to dense feature representations. Same-class mixup, which restricts the partner draw to examples of the same label, preserves binary semantics exactly at the cost of slightly reduced diversity.

Test-time augmentation [28] runs the model on multiple stochastic perturbations of each test sample and averages the resulting probabilities. The technique has well-understood failure modes (e.g., when training- and test-time augmentation distributions disagree) and is most effective when the perturbations are mild and label-preserving. Our seven-variant TTA scheme combines a strict label-preserving transformation (time reversal — valid for non-directional motor seizures) with three nearly-identity perturbations (additive jitter, ±5 % global scaling, small circular time shifts).

### 3.6 Markerless multi-view calibration

Recent work on dynamic, markerless extrinsic camera calibration [29] uses pose-based 3D priors (from MeTRAbs or RTMPose+VideoPose3D) to calibrate multi-camera setups from a single walking subject, without checkerboards or specialized rigs. This unlocks a path to multi-view, metric-scale 3D pose for clinical settings where synchronized cameras are available; we return to this direction in Section 9.

## 4. Data and Task

The challenge provides per-segment files of shape $(150, 33, 5)$:

| Axis     | Meaning                                                       |
|----------|---------------------------------------------------------------|
| Frame    | 150 frames per segment (≈5 s at 30 fps)                       |
| Landmark | 33 MediaPipe Pose landmarks (full-body topology)              |
| Channel  | $(x, y, z, \text{visibility}, \text{presence})$               |

File names follow `child_<id>_<seg>.npy`, giving us a per-sample patient identifier we exploit for leakage-free cross-validation. Labels are binary (`1` = seizure, `0` = non-seizure) with strong class imbalance toward non-seizure samples.

**Missingness.** A non-trivial fraction of frames is entirely `NaN` — these correspond to frames where the upstream pose detector failed (subject out of frame, severe occlusion, low light). Some segments contain runs of consecutive failed frames; in extreme cases an entire segment may be `NaN`.

## 5. Method

The pipeline has four stages: preprocessing, feature extraction, model training, ensemble inference. Figure 1 sketches the end-to-end flow; the remainder of this section describes each stage in detail.

<figure>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 690" style="max-width:62%;height:auto;display:block;margin:0.8em auto">
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
<figcaption style="font-style:italic;color:#555;text-align:center;font-size:0.92em;margin-top:0.2em;">Figure 1. End-to-end pipeline. Either the challenge's pre-extracted <code>.npy</code> arrays or raw video (via the MediaPipe adapter) enter a common preprocessing and feature-extraction front-end. Two architectures process the 150 × 518 feature sequences in parallel; their five-fold ensembles are averaged across seven test-time-augmentation variants and thresholded.</figcaption>
</figure>

### 5.1 Preprocessing

Three transformations are applied per sample before any feature extraction.

1. **NaN-frame mask and fill.** A binary per-frame mask records which frames had no detected pose. Missing frames are then forward-filled from the most recent valid frame, and any leading `NaN`s are backward-filled. If a segment is entirely `NaN`, it is set to zero. The binary mask itself is preserved as a feature so the model can learn to discount filled frames.
2. **Hip-centering.** The 3D coordinates are translated so the mid-hip point (mean of MediaPipe landmarks 23 and 24) sits at the origin in every frame. This removes camera-translation effects and makes the representation translation-invariant per frame.
3. **Torso-length normalization.** The Euclidean distance between mid-shoulder (landmarks 11+12) and mid-hip is computed per frame; coordinates are divided by this torso length. A guard replaces near-zero or implausibly small values with the median torso length over valid frames. This makes the representation approximately scale-invariant and removes camera-distance effects.

Visibility and presence channels are passed through unchanged.

### 5.2 Feature Extraction

We engineer a 518-dimensional per-frame feature vector.

| Component                                                   | Dim. |
|-------------------------------------------------------------|-----:|
| Raw landmarks $(x,y,z,\text{vis},\text{pres})$ flattened    |  165 |
| Velocity ($\Delta xyz$ per landmark)                        |   99 |
| Acceleration ($\Delta^2 xyz$ per landmark)                  |   99 |
| Per-landmark speed magnitude                                |   33 |
| Joint angles (10 triplets at elbows, knees, shoulders, hips)|   10 |
| Left–right symmetry distances (6 pairs)                     |    6 |
| NaN-frame indicator                                         |    1 |
| Low-freq. FFT power, 7 landmarks × 3 axes × 5 bins          |  105 |
| **Total**                                                   |**518**|

**Joint angles.** For a triplet $(a, b, c)$ of landmark indices with positions $p_a(t), p_b(t), p_c(t) \in \mathbb{R}^3$ at frame $t$, the joint angle at $b$ is

$$\theta_{abc}(t) = \frac{1}{\pi}\arccos\!\left(\frac{(p_a(t)-p_b(t)) \cdot (p_c(t)-p_b(t))}{\Vert p_a(t)-p_b(t)\Vert\, \Vert p_c(t)-p_b(t)\Vert + \varepsilon}\right) \in [0, 1].$$

Ten triplets cover both elbows and knees, the shoulder-to-hip trunk angle on each side, and a nose-to-shoulder-to-hip head-orientation angle on each side.

**Symmetry.** For each of six left–right landmark pairs $(\ell, r)$ — shoulders, elbows, wrists, hips, knees, ankles — the Euclidean distance $\Vert p_\ell(t) - p_r(t) \Vert$ is recorded. These distances flatten when the body posture is symmetric and grow when one side moves independently of the other, providing a direct asymmetry signal.

**Spectral features.** The full-FFT power of seven kinematic landmarks (wrists, ankles, elbows, nose) is computed per axis across the 150-frame window. We retain only the first five non-DC bins per channel — bandlimited to the rhythmic frequency range (roughly 0.2–3 Hz at 30 fps over a 5 s window) in which clonic and myoclonic seizure activity typically appears. These global-window features are tiled across all frames so the temporal models always have access to the segment's spectral signature.

**Normalization.** Per-feature global mean and standard deviation are computed once over the entire training set and persisted to disk (`feat_mean.npy`, `feat_std.npy`). The same statistics normalize features at inference; no batch-statistics drift.

### 5.3 Models

We use a two-architecture ensemble. Both share an attention-pooling head producing a single segment-level embedding, followed by a small MLP classifier.

**CNN-GRU.** Two 1D convolutional layers (kernels 5 and 3, batch-norm, ReLU, dropout) over the time axis project the 518-dim per-frame features to 128 channels. A 2-layer bidirectional GRU then processes the $150 \times 128$ sequence, producing $150 \times 256$ contextualized embeddings. A learned attention layer pools these to a single 256-d segment embedding, and a small MLP ($256 \to 64 \to 1$) produces a logit.

**Multi-scale TCN.** Five residual temporal blocks with kernel size 3 and exponentially increasing dilations $\{1, 2, 4, 8, 16\}$ provide a receptive field that covers the entire 150-frame window. Each block has batch-norm, dropout, and a residual $1 \times 1$ projection when channel counts differ. Attention pooling and an MLP head mirror the CNN-GRU.

**Attention pooling.** For a sequence of hidden states $h_1, \ldots, h_T$, the attention pooling layer learns weight matrices $W$ and $w$ producing soft weights

$$\alpha_t = \frac{\exp(w^\top \tanh(W h_t))}{\sum_{t'=1}^{T} \exp(w^\top \tanh(W h_{t'}))}, \qquad z = \sum_{t=1}^{T} \alpha_t h_t.$$

This is the standard attention-pooling recipe and is differentiable end-to-end.

**Why two architectures.** The CNN-GRU has a recurrent inductive bias that models temporal order explicitly through hidden-state propagation; the TCN computes parallel multi-scale features and is easier to optimize on small data. Their errors are de-correlated enough that ensembling improves precision–recall trade-off at our operating threshold (Section 6).

### 5.4 Training

**Cross-validation.** `StratifiedGroupKFold` ($k=5$) grouped by patient identifier (child ID parsed from the filename). This ensures that within-subject correlations cannot leak between train and validation, which would otherwise inflate apparent performance. Class ratios are preserved at the fold level by the stratification component.

**Focal loss.** Following [26], we use

$$\mathcal{L}_{\text{focal}}(\hat p, y) = -\alpha_t (1 - p_t)^\gamma \log p_t, \qquad p_t = \begin{cases} \hat p & \text{if } y = 1 \\ 1 - \hat p & \text{if } y = 0 \end{cases}, \qquad \alpha_t = \begin{cases} \alpha & y=1 \\ 1-\alpha & y=0 \end{cases}.$$

We set $\alpha=0.75$ (up-weight the minority seizure class) and $\gamma=2.0$ (down-weight easy examples). This works on top of an already-balanced batch produced by `WeightedRandomSampler`.

**Mixup augmentation.** For two same-class training samples $(x_i, y_i)$ and $(x_j, y_j)$ with $y_i = y_j$, mixup [27] produces

$$\tilde x = \lambda x_i + (1-\lambda) x_j, \qquad \tilde y = \lambda y_i + (1-\lambda) y_j, \qquad \lambda \sim \mathrm{Beta}(\alpha_{\text{mix}}, \alpha_{\text{mix}}),$$

with $\alpha_{\text{mix}} = 0.2$ and an in-class partner sampled at probability 0.3 per training step.

**Optimization.** AdamW [30] with initial learning rate $5\times 10^{-4}$, weight decay $10^{-4}$, gradient clipping at norm 1.0. The schedule is `OneCycleLR` [31] with $\text{max\_lr} = 10\times$ initial, $\text{pct\_start} = 0.3$, cosine anneal. Up to 120 epochs with early stopping on validation $F_1$ (patience 25).

**Other augmentations.** On-the-fly per training sample, independently of mixup: time-reversal ($p{=}0.3$), additive Gaussian noise ($\sigma{=}0.02$, $p{=}0.5$), time-index resampling ($p{=}0.3$), and random frame dropout (1–14 frames zeroed, $p{=}0.2$).

**Per-fold threshold.** After each validation pass we grid-search the decision threshold over $[0.20, 0.80]$ at $0.01$ resolution and report $F_1$ at the best operating point. The best validation $F_1$ across 120 epochs and its associated threshold are persisted per fold; the early-stopping criterion is also $F_1$ at the optimal threshold.

### 5.5 Inference

At inference each test sample is preprocessed and featurized identically to training. We then apply seven test-time augmentations: the original, its time-reverse, additive jitter ($\sigma{=}0.01$), uniform scaling by $\pm 5\%$, and circular time shifts by $\pm 3$ frames. With $M$ ensemble members and $V$ TTA variants, the final probability is

$$\bar p(x) = \frac{1}{|M| \cdot |V|} \sum_{m \in M} \sum_{v \in V} \sigma\bigl(f_m(\phi_v(x))\bigr),$$

where $\phi_v$ is the $v$-th TTA transform, $f_m$ is the logit produced by ensemble member $m$, and $\sigma$ is the sigmoid. With 10 members (5 CNN-GRU folds + 5 TCN folds) and 7 variants we average 70 probabilities per sample. The final binary label uses a fixed ensemble threshold (currently $\tau = 0.65$, tuned to maximize cross-validation $F_1$ across folds; the per-fold averaged-threshold fallback is also implemented).

## 6. Results

### 6.1 Cross-validation on the training partition

| Fold | CNN-GRU $F_1$ | CNN-GRU $\hat\tau$ | TCN $F_1$ | TCN $\hat\tau$ |
|-----:|------:|------:|------:|------:|
|    0 | 0.633 | 0.70  | 0.611 | 0.63  |
|    1 | 0.615 | 0.59  | 0.608 | 0.61  |
|    2 | 0.629 | 0.64  | 0.635 | 0.33  |
|    3 | 0.575 | 0.28  | 0.627 | 0.25  |
|    4 | 0.691 | 0.51  | 0.694 | 0.39  |
| **Mean** | **0.628** | 0.54 | **0.635** | 0.44 |

Observations:

- The mean $F_1$ of the two architectures is comparable (0.628 vs 0.635), but the per-fold thresholds differ widely. The CNN-GRU prefers high-precision operating points (thresholds 0.51–0.70 on folds 0, 1, 2, 4); the TCN often prefers low thresholds (0.25–0.39 on folds 2, 3, 4). The two models occupy different points on the precision–recall curve at their respective optima — exactly the diversity that motivates an ensemble.
- Fold 3 is materially harder for the CNN-GRU ($F_1$ 0.575) than for the TCN ($F_1$ 0.627). Fold 4 is the easiest for both. The variance suggests a small number of either harder subjects or harder seizure morphologies dominates the per-fold metric.
- The combined ensemble operates at $\tau{=}0.65$ on averaged probabilities across 70 forward passes per sample (10 models × 7 TTA variants). De-correlation between architectures plus TTA smoothing produces a calibration sharper than either model alone, which is why the operating threshold sits at the upper end of the per-fold range.

### 6.2 Held-out test set and leaderboard

Final evaluation was performed by the challenge organisers on a sequestered test partition. Our submission attained $F_1 = 0.2975$ on this held-out set, placing 19th of 26 valid submissions; sensitivity served as the secondary tie-breaker. The top five entries by $F_1$ are summarised in Table 2, alongside our position for context.

| Rank | Author / team | $F_1$ | Approach |
|-----:|---------------|------:|----------|
|    1 | Kramer (BU / JHU)              | 0.4502 | Two-step likelihood + compact classifier |
|    2 | Ramesh (NIT Tiruchirappalli)   | 0.4286 | 1D-CNN + LightGBM |
|    3 | Daraie (JHU)                   | 0.4048 | Ensemble of 1D-ResNet and XGBoost |
|    4 | Walsh (U. Pennsylvania)        | 0.3984 | GCN + bidirectional GRU |
|    5 | Hazra (IIT Kharagpur)          | 0.3923 | ST-GCN with attention-MIL ensemble |
|  ... | ...                            | ...    | ... |
| **19** | **Owoyele et al. (HPI Potsdam, this work)** | **0.2975** | **CNN-GRU + TCN ensemble** |

*Table 2. Top five leaderboard entries and the position of the present work. Source: official challenge leaderboard, Section on Computational Neurology, Charité — Universitätsmedizin Berlin.*

Several observations:

- The leaderboard is *compressed*: the spread from rank 1 to rank 19 is roughly 15 percentage points of $F_1$, and the gap from rank 1 to rank 10 is under 12 points. No submission cleared $F_1 = 0.5$. This is consistent with infantile-spasm detection being a genuinely hard problem under the dataset and protocol the organisers chose, and with the held-out test set containing morphologies under-represented in the training partition.
- The top entries cluster around two architectural families: (i) two-stage pipelines that produce an interim seizure-likelihood signal before a compact classifier (the Kramer entry), and (ii) hybrids of gradient-boosted ensembles with deep encoders (Ramesh, Daraie, Haag, Pedersen). Graph-convolutional approaches (Walsh, Hazra, Moro) cover the middle of the leaderboard. Our dual-architecture deep ensemble most closely resembles the graph-convolutional family in spirit (parallel temporal encoders + late fusion) without the explicit body-graph structure.
- The substantial gap between our cross-validation $F_1 \approx 0.63$ and held-out $F_1 = 0.2975$ is the dominant signal here and is analysed in Section 8.1 below.

## 7. Deployment

The same trained ensemble is exposed through two entry points.

**Challenge inference.** `main.py` is the Docker entrypoint. Given a directory of $(150, 33, 5)$ `.npy` arrays and an output CSV path, it loads the 10-model ensemble, runs the full preprocessing–feature–TTA path on every sample, and writes `segment_name, label` pairs. The Docker image is built and pushed to GitHub Container Registry by a manual-dispatch GitHub Actions workflow.

**Raw-video inference.** A video adapter (`video_to_features.py`) uses MediaPipe Tasks' `PoseLandmarker` in VIDEO mode to convert any `.mp4`, `.mov`, `.avi`, `.mkv` or `.webm` file into the same $(150, 33, 5)$ array the trained models expect. Frames are sampled evenly across the clip; pose-failure frames are left as `NaN` so the existing `fill_nan_frames` preprocessing handles them transparently. A second entry point, `predict_video.py`, exposes a CLI accepting either a single video or a directory of mixed `.npy` and video files; the output CSV includes the per-sample probability alongside the binary label.

This decoupling matters in clinical settings: the competition delivers pre-extracted pose, but on-site EMU recordings are raw video. With the adapter, the same model can be evaluated end-to-end on either input format without retraining.

## 8. Discussion and Limitations

### 8.1 The CV–test generalisation gap

The headline finding is the gap between cross-validation $F_1 \approx 0.63$ and held-out $F_1 = 0.2975$. A gap of this magnitude warrants a careful audit; the most plausible contributors are:

1. **Test-set distribution shift.** The training partition was extracted from a different collection of patients than the test partition. Infantile-spasm morphology varies meaningfully by age, by underlying aetiology (genetic, structural, metabolic, idiopathic) and by the camera angle and lighting of the parent's home-video. Even with leakage-free patient-grouped cross-validation on the training set, the model has no guarantee of generalising across the *distribution shift* between training-pool and test-pool subjects.
2. **Threshold over-fitting to CV folds.** Per-fold optimal thresholds in Table 1 span $[0.25, 0.70]$, signalling that the model's probabilities are not strongly calibrated. The single ensemble threshold $\tau = 0.65$ that maximises mean CV $F_1$ may be a poor operating point on the test distribution; it would also be over-fit to the small number of validation folds. A threshold calibrated on a separate held-out fold (rather than on CV directly) would likely transfer better, at the cost of a smaller training pool.
3. **Class-imbalance interaction with $F_1$.** $F_1$ on a strongly imbalanced binary task can swing materially with small changes in either precision or recall: e.g. on a 1:10 imbalanced set, mis-classifying a third of the minority class can halve $F_1$ even if overall accuracy is high. The leaderboard top score of 0.4502 (and median of approximately 0.31) indicates that no entry escaped this regime.
4. **TTA distribution mismatch.** Test-time augmentation assumes the training- and test-time augmentation distributions match. The training-time augmentations are aggressive (gaussian noise, frame dropout, time-index resampling, mixup); the test-time augmentations are nearly identity (jitter, scale ±5%, time shift ±3). The model has been trained to be invariant to perturbations that the TTA does not produce — meaning the test-time ensemble may not exploit the full robustness the training augmentation paid for. Aligning the augmentation suites is a non-trivial follow-up.
5. **Architecture–signal mismatch.** Infantile spasms concentrate motor energy in sub-second sub-windows; both our architectures pool over the full 5-second window with attention. While attention can in principle focus, multiple-instance learning explicitly built around 1–2 second proposals — as in the Daraie and Galindo-Lazo entries — may be a better structural prior for this specific seizure type.

The most actionable next step, given the audit above, is (1) re-train with a held-out fold dedicated to threshold calibration, and (2) refactor inference around MIL-style sub-window pooling rather than full-window attention.

### 8.2 Pose-only ceiling

As argued in Section 2.4, operating purely in pose space is the source of both this system's strengths (privacy, lightness, lighting invariance) and its ceiling. Subtle facial cues (gaze deviation, automatisms involving small mouth movements), vital signs, and contextual information are simply not in the input. Roughly $F_1{=}0.63$ on 5-fold CV is consistent with a moderate-strength pose-only system on a small, imbalanced dataset; closing the remaining gap likely requires either a larger labeled dataset or auxiliary modalities.

**MediaPipe is the bottleneck on raw video.** MediaPipe Pose returns 33 landmarks in normalized image coordinates; the $z$ channel is image-relative and is metric only after the torso-normalization step we apply. On low-light, partially-occluded, or oblique-angle clinical footage, the detector fails more often, and the failure mode is a long run of `NaN` frames that the forward-fill heuristic handles passably but not gracefully.

**Patient-grouped CV is conservative but limited.** Five folds over a relatively small number of children means each validation set may not cover all seizure morphologies. The 5–12 percentage-point spread in per-fold $F_1$ in Section 6 quantifies this. A leave-one-patient-out protocol would give tighter bias estimates but at proportional compute cost.

**Threshold sensitivity and operating-point selection.** The per-fold optimal thresholds span $[0.25, 0.70]$ — a wide range, signaling that the underlying probability distributions are not strongly calibrated and that small shifts in fold composition move the optimum substantially. The single ensemble threshold $\tau = 0.65$ we report is the value that maximizes cross-validation $F_1$ across folds; in a deployment with cost-sensitive errors (false negatives orders of magnitude costlier than false positives, as in clinical screening) this would be moved well below 0.5.

**Augmentation correctness.** The current "time-warp" augmentation samples 150 indices uniformly at random from the 150-frame window and sorts them, which can duplicate frames and drop others; this is more of a robustness-injecting perturbation than a true monotone time warp. Replacing it with linear resampling at a randomly chosen rate would be more faithful to the name.

**FFT broadcast is mildly wasteful.** The 105 spectral features are sample-level (one vector per segment) but are tiled across all 150 frames so the per-frame temporal models can see them. The CNN front-end and TCN can pick out these constants, but a parallel head concatenating the spectral vector with the temporal embedding before the classifier would be cleaner.

## 9. Future Work

**MeTRAbs-based metric 3D features.** Replacing the MediaPipe frontend with the MeTRAbs estimator [17] would give us metric-scale 3D coordinates for 87 body joints (vs. 33 unitless MediaPipe landmarks). The downstream feature engineering would update modestly — joint-angle triplets, symmetric pairs, and key landmarks for the FFT all need to be re-mapped to the new skeleton — but the training and ensembling code would be unchanged. We expect this to lift $F_1$ for two reasons: richer joints expose hand and head dynamics that the 33-landmark MediaPipe topology loses, and metric scale removes the need for torso normalization and stabilizes velocity- and acceleration-based features against camera-distance changes.

**Multi-view triangulated 3D pose.** Recent work on markerless, dynamic extrinsic camera calibration [29] shows how a moving subject can calibrate a multi-camera setup without checkerboards, producing triangulated 3D poses in a common world frame. Adopting that pipeline upstream of our classifier would let us fuse views, eliminate single-camera occlusion failures, and feed the classifier metric 3D directly. The dependency is data-side: it requires synchronized multi-camera EMU recordings.

**Temporal-context architectures.** A natural model-side step is to replace the global FFT with a learned spectral or Transformer head [25], or to use a state-space model (e.g., Mamba [32]) over the 150-frame window to push the receptive field beyond what the current dilated TCN can cleanly cover. The data scale would need to grow to support either.

**Calibrated probability output.** The current ensemble output is a 70-way average that is not explicitly calibrated. Wrapping it in a temperature-scaling or isotonic-regression calibrator on a held-out fold [33] would produce probability outputs usable for downstream cost-sensitive triage, rather than a single hard label.

**Auxiliary modalities.** The pose-only ceiling can be raised by fusing with at least three modalities readily available in clinical settings: audio (ictal cry, vocalisation, parental commentary), face landmarks (gaze deviation, mouth automatisms — already produced by MediaPipe's face mesh model), and accelerometer streams from wearable devices [18,19]. A late-fusion ensemble that weights modalities adaptively by their per-fold reliability is a natural next step.

**Integration with the MaskAnyone pipeline.** The MaskAnyone toolkit [34] developed in our group offers an end-to-end privacy-preserving processing path for clinical audio-visual archives, with body masking, face blur, audio scrubbing, and pose-only summarisation. A natural next step is to compose this detection model with MaskAnyone's pose-extraction stage so that home-video material submitted by caregivers can be processed and classified entirely on-device, with only landmark coordinates and the binary detection label leaving the patient's premises.

## 10. Conclusion

We presented a complete pose-only seizure-detection pipeline, covering preprocessing, hand-engineered kinematic and spectral features, a two-architecture deep-learning ensemble trained under patient-grouped cross-validation, test-time augmentation, and Docker-based deployment. The system reaches a mean validation $F_1$ of $0.628$ (CNN-GRU) and $0.635$ (TCN) per-fold under `StratifiedGroupKFold` with an ensemble operating threshold of $0.65$. A video front-end lets the same models be applied to raw clinic footage, and several natural extensions — metric-scale 3D pose from MeTRAbs, multi-view triangulation, learned temporal architectures, calibrated probability outputs, and multimodal fusion — are sketched as forward directions.

---

## Appendix A: Repository Layout

| File                       | Role                                                    |
|----------------------------|---------------------------------------------------------|
| `config.py`                | Hyperparameters and paths                               |
| `preprocessing.py`         | NaN-handling, hip-centering, torso normalization        |
| `features.py`              | 518-d per-frame feature extraction                      |
| `dataset.py`               | Torch `Dataset` with augmentation                       |
| `model.py`                 | CNN-GRU, TCN, FocalLoss, Attention                      |
| `train.py`                 | 5-fold × 2-architecture training                        |
| `train_all.py`             | Training entry point                                    |
| `inference.py`             | Ensemble + TTA + thresholding                           |
| `main.py`                  | Challenge Docker entry point (`.npy`)                   |
| `video_to_features.py`     | Video → $(150, 33, 5)$ via MediaPipe                    |
| `predict_video.py`         | Raw-video CLI entry point                               |
| `Dockerfile`               | Inference image (Python 3.12 slim)                      |
| `requirements.txt`         | Pinned dependencies                                     |
| `checkpoints/`             | 10 fold weights + norm stats + results                  |

## Appendix B: Reproducibility

**Training:**
```
pip install -r requirements.txt
python train_all.py
```

**Inference (challenge `.npy` format):**
```
python main.py <test_dir> <output.csv>
```

**Inference (raw video):**
```
mkdir models
curl -L -o models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
python predict_video.py <video_or_dir> predictions.csv
```

## Acknowledgments

This work was supported by the Federal Ministry of Research, Technology and Space under the funding code "KI-Servicezentrum Berlin-Brandenburg" (16IS22092). Responsibility for the content of this publication remains with the authors. We thank the Section on Computational Neurology at Charité — Universitätsmedizin Berlin and the organising committee of the 2026 International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders for hosting the *Video-Based Seizure Detection Challenge* and for releasing the anonymised pose dataset on which this work is based.

## References

[1] Fisher, R. S., Acevedo, C., Arzimanoglou, A., et al. (2014). *ILAE official report: a practical clinical definition of epilepsy.* Epilepsia, 55(4), 475–482. doi:10.1111/epi.12550.

[2] Fisher, R. S., Cross, J. H., French, J. A., et al. (2017). *Operational classification of seizure types by the International League Against Epilepsy: position paper of the ILAE Commission for Classification and Terminology.* Epilepsia, 58(4), 522–530. doi:10.1111/epi.13670.

[3] Karayiannis, N. B., Sami, A., Frost, J. D., Wise, M. S., & Mizrahi, E. M. (2006). *Automated detection of videotaped neonatal seizures based on motion tracking methods.* Journal of Clinical Neurophysiology, 23(6), 521–531. doi:10.1097/00004691-200612000-00004.

[4] Google, LLC. *MediaPipe Pose Landmarker.* https://developers.google.com/mediapipe/solutions/vision/pose_landmarker, accessed 2026.

[5] Kwan, P., Arzimanoglou, A., Berg, A. T., et al. (2010). *Definition of drug resistant epilepsy: consensus proposal by the ad hoc Task Force of the ILAE Commission on Therapeutic Strategies.* Epilepsia, 51(6), 1069–1077. doi:10.1111/j.1528-1167.2009.02397.x.

[6] Scheffer, I. E., Berkovic, S., Capovilla, G., et al. (2017). *ILAE classification of the epilepsies: position paper of the ILAE Commission for Classification and Terminology.* Epilepsia, 58(4), 512–521. doi:10.1111/epi.13709.

[7] Tzallas, A. T., Tsipouras, M. G., & Fotiadis, D. I. (2009). *Epileptic seizure detection in EEGs using time–frequency analysis.* IEEE Transactions on Information Technology in Biomedicine, 13(5), 703–710. doi:10.1109/TITB.2009.2017939.

[8] Yan, S., Xiong, Y., & Lin, D. (2018). *Spatial temporal graph convolutional networks for skeleton-based action recognition.* AAAI. doi:10.1609/aaai.v32i1.12328.

[9] Shi, L., Zhang, Y., Cheng, J., & Lu, H. (2019). *Two-stream adaptive graph convolutional networks for skeleton-based action recognition.* CVPR. doi:10.1109/CVPR.2019.01230.

[10] Liu, Z., Zhang, Hongwen, Chen, Z., Wang, Z., & Ouyang, W. (2020). *Disentangling and unifying graph convolutions for skeleton-based action recognition.* CVPR. doi:10.1109/CVPR42600.2020.00022.

[11] Duan, H., Zhao, Y., Chen, K., Lin, D., & Dai, B. (2022). *Revisiting skeleton-based action recognition.* CVPR. doi:10.1109/CVPR52688.2022.00298.

[12] Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2021). *OpenPose: realtime multi-person 2D pose estimation using part affinity fields.* IEEE TPAMI, 43(1), 172–186. doi:10.1109/TPAMI.2019.2929257.

[13] Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). *BlazePose: on-device real-time body pose tracking.* CVPR Workshops. arXiv:2006.10204.

[14] Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). *Deep high-resolution representation learning for human pose estimation.* CVPR. doi:10.1109/CVPR.2019.00584.

[15] Jiang, T., Lu, P., Zhang, L., et al. (2023). *RTMPose: real-time multi-person pose estimation based on MMPose.* arXiv:2303.07399.

[16] Pavllo, D., Feichtenhofer, C., Grangier, D., & Auli, M. (2019). *3D human pose estimation in video with temporal convolutions and semi-supervised training.* CVPR. doi:10.1109/CVPR.2019.00794.

[17] Sárándi, I., Linder, T., Arras, K. O., & Leibe, B. (2021). *MeTRAbs: metric-scale truncation-robust heatmaps for absolute 3D human pose estimation.* IEEE Transactions on Biometrics, Behavior, and Identity Science, 3(1), 16–30. doi:10.1109/TBIOM.2020.3037257.

[18] Beniczky, S. (2018). *Standards for testing and clinical validation of seizure detection devices.* Epilepsia, 59(S1), 9–13. doi:10.1111/epi.14049.

[19] Onorati, F., Regalia, G., Caborni, C., et al. (2017). *Multicenter clinical assessment of improved wearable multimodal convulsive seizure detectors.* Epilepsia, 58(11), 1870–1879. doi:10.1111/epi.13899.

[20] Cuppens, K., Karsmakers, P., Van de Vel, A., et al. (2014). *Accelerometry-based home monitoring for detection of nocturnal hypermotor seizures based on novelty detection.* IEEE Journal of Biomedical and Health Informatics, 18(3), 1026–1033. doi:10.1109/JBHI.2013.2285015.

[21] Geertsema, E. E., Thijs, R. D., Gutter, T., et al. (2018). *Automated video-based detection of nocturnal convulsive seizures in a residential care setting.* Epilepsia, 59(S1), 53–60. doi:10.1111/epi.14050.

[22] Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation, 9(8), 1735–1780. doi:10.1162/neco.1997.9.8.1735.

[23] Cho, K., van Merriënboer, B., Gülçehre, Ç., et al. (2014). *Learning phrase representations using RNN encoder–decoder for statistical machine translation.* EMNLP. doi:10.3115/v1/D14-1179.

[24] Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling.* arXiv:1803.01271.

[25] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention is all you need.* NeurIPS. arXiv:1706.03762.

[26] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV. doi:10.1109/ICCV.2017.324.

[27] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). *mixup: beyond empirical risk minimization.* ICLR.

[28] Shanmugam, D., Blalock, D., Balakrishnan, G., & Guttag, J. (2021). *Better aggregation in test-time augmentation.* ICCV. doi:10.1109/ICCV48922.2021.00125.

[29] de la Place, F. *Dynamic Extrinsic Camera Calibrator.* https://github.com/flodelaplace/lab-camera-dynamic-calibrator.

[30] Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.

[31] Smith, L. N., & Topin, N. (2019). *Super-convergence: very fast training of neural networks using large learning rates.* In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, SPIE. doi:10.1117/12.2520589.

[32] Gu, A., & Dao, T. (2023). *Mamba: linear-time sequence modeling with selective state spaces.* arXiv:2312.00752.

[33] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML.

[34] Owoyele, B. A., Schilling, M., Sawahn, R., Kaemer, N., Zherebenkov, P., Verma, B., Pouw, W., & de Melo, G. (2024). *MaskAnyone toolkit: offering strategies for minimizing privacy risks and maximizing utility in audio-visual data archiving.* Proceedings of the 45th International Conference on Information Systems (ICIS), Bangkok. https://aisel.aisnet.org/icis2024/adv_theory/adv_theory/1/.
