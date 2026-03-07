# Human Activity Recognition Using Hidden Markov Models

## Report

### 1. Background and Motivation

Human Activity Recognition (HAR) has become important for health monitoring, fitness tracking, and smart home systems. With smartphones having built-in accelerometers and gyroscopes, we can capture motion data without extra hardware. In this project, we use sensor data from an iPhone 13 to classify four activities -- standing, walking, jumping, and still -- using a Hidden Markov Model. We chose HMMs because they give us interpretable transition probabilities between activities, which matters for understanding behavior patterns (for example in physical rehabilitation where knowing *how* someone transitions between activities is as important as identifying them).

### 2. Data Collection and Preprocessing

#### 2.1 Collection Setup

We used the Sensor Logger app (v1.54) on an iPhone 13 running iOS 18.6.2. Both accelerometer and gyroscope data were recorded at 100 Hz (10ms intervals). Each recording lasted about 5-10 seconds. Data was collected by one participant performing each activity in a controlled setting.

| Parameter | Value |
|-----------|-------|
| Device | iPhone 13 |
| App | Sensor Logger v1.54 |
| Sampling Rate | 100 Hz |
| Sensors | Accelerometer (x,y,z) + Gyroscope (x,y,z) |

#### 2.2 Dataset

| Activity | Recordings | Approx. Data Points |
|----------|-----------|---------------------|
| Standing | 8 | ~6,200 |
| Walking | 10 | ~8,600 |
| Jumping | 10 | ~10,600 |
| Still | 10 | ~8,000 |
| **Total** | **38** | **~33,400** |

For standing, the phone was held at waist level. Walking was at a normal pace. Jumping was continuous vertical jumps. For still, the phone was placed flat on a table.

#### 2.3 Preprocessing

The app exports each recording as a folder with separate CSVs per sensor. We loaded `Accelerometer.csv` and `Gyroscope.csv` from each folder and merged them on nearest timestamp (since the sensors aren't perfectly synchronized).

From each merged recording, we extracted **108 features** (72 time-domain + 36 frequency-domain):

**Time-domain (72 features):**
- **Mean / Median** per axis: captures baseline orientation (e.g. gravity direction when standing)
- **Variance / Std dev**: measures signal fluctuation -- high for jumping, near-zero for still
- **Min / Max / Range**: captures amplitude of motion -- jumping has massive range vs still
- **RMS**: overall signal energy, separates active from inactive states
- **Skewness / Kurtosis**: distribution shape -- asymmetric walking gait produces skewed signals
- **SMA (Signal Magnitude Area)**: total acceleration across all axes, general activity level indicator
- **Magnitude std/range**: variability of the combined 3-axis magnitude
- **Inter-axis correlations (XY, XZ, YZ)**: how axes co-move -- walking has correlated hip sway while jumping is mostly vertical (high Z-axis, low XY correlation)

**Frequency-domain (36 features):**
- **Dominant frequency**: main repetition rate -- walking has ~2 Hz step cycle, jumping ~1-3 Hz, still has none
- **Spectral energy**: total power in frequency spectrum -- more energy means more dynamic activity
- **First 3 FFT components**: magnitudes of the strongest frequency peaks
- **Spectral entropy**: how spread the energy is across frequencies -- walking is concentrated (low entropy) while random noise is spread out (high entropy)

**Normalization:** We applied Z-score standardization (zero mean, unit variance) using `StandardScaler` so that features measured in different units (m/s² for acceleration vs rad/s for angular velocity) get equal weight in the model. Without this, features with larger raw values would dominate the HMM's Gaussian emission distributions.

### 3. HMM Setup and Implementation

#### 3.1 Model Design

We used a classification approach: train one Gaussian HMM per activity, then classify new samples by checking which model gives the highest log-likelihood. Each model has 2 internal sub-states (we tried 3 but it caused convergence issues with our dataset size).

| Component | What it represents |
|-----------|-------------------|
| Hidden states (Z) | Standing, Walking, Jumping, Still |
| Observations (X) | 108-dim feature vectors |
| Transition matrix (A) | Sub-state transitions within each activity model |
| Emission model (B) | Gaussian with diagonal covariance per sub-state |
| Initial probs (π) | Uniform across sub-states |

#### 3.2 Training (Baum-Welch)

We used `hmmlearn`'s GaussianHMM which runs Baum-Welch (EM) internally. The convergence criterion was set to `tol=1e-4` -- meaning training stops when the log-likelihood improvement between EM iterations falls below 0.0001, indicating the model parameters have stabilized. Maximum iterations was set to 200 as a safety limit.

All four activity models converged successfully within the iteration limit.

#### 3.3 Viterbi Decoding

We implemented the Viterbi algorithm from scratch to decode activity sequences. We defined a 4×4 transition matrix with high self-transition probabilities (0.70) reflecting the fact that people tend to continue their current activity. The emission parameters (mean and variance per state) were computed from the training data's feature vectors.

### 4. Results

#### 4.1 Classification

| Set | Accuracy |
|-----|----------|
| Training | 100.0% |
| Test | 100.0% |

#### 4.2 Per-Activity Metrics (Test Set)

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|----------|---------|-------------|-------------|----------|
| Standing | 2 | 1.0000 | 1.0000 | 1.0000 |
| Walking | 2 | 1.0000 | 1.0000 | 1.0000 |
| Jumping | 3 | 1.0000 | 1.0000 | 1.0000 |
| Still | 3 | 1.0000 | 1.0000 | 1.0000 |

The 100% accuracy is helped by the fact that our dataset is small and the four activities produce quite different sensor patterns. A larger test set or more similar activities would likely lower these numbers.

#### 4.3 Viterbi Decoding

The manual Viterbi tested on a sequence of: still → standing → walking → jumping → standing → still. It decoded all 6 steps correctly.

#### 4.4 Visualizations

The notebook includes:
- Raw accelerometer and gyroscope signal plots
- Signal magnitude box plots across activities
- Feature distribution histograms
- Learned transition matrix heatmaps for each activity model
- **Emission probability distributions** (Gaussian PDFs per sub-state per activity)
- Training and test confusion matrices
- Decoded activity sequence plot (Viterbi true vs predicted)
- Per-activity evaluation metrics bar chart

### 5. Discussion and Conclusion

**Easiest activities to tell apart:** Jumping was easiest -- very high amplitude periodic pattern that's nothing like the others. Still was also easy since both sensors show near-zero variance. The hardest pair is standing vs still, since both have low movement, but body sway when holding the phone at waist level (standing) vs placing it on a table (still) gives enough of a difference.

**Transition matrix:** The high diagonal values (0.70) reflect that people usually continue doing what they're doing rather than constantly switching. Standing-walking transition probability is higher since those naturally follow each other. Jumping has low transition probabilities since it's a deliberate, short-duration activity.

**Sensor noise and sampling rate:** 100 Hz is more than sufficient -- human movement is mostly below 20 Hz, so we capture everything with room to spare. Some noise is visible in still/standing recordings but computing statistics over the recording window smooths this out. Both accelerometer and gyroscope were useful; the gyroscope captures rotational patterns during walking that the accelerometer alone might miss.

**What we'd improve with more time:**
- Collect more recordings, especially standing (only 8 vs 10 for others)
- Use sliding windows to get more training samples from each recording
- Try adding magnetometer data for orientation
- Run PCA or feature selection to reduce the 108 features
- Test with data from other people to assess generalization
- The 100% accuracy is somewhat expected given the small test set -- a larger evaluation would give more confidence
