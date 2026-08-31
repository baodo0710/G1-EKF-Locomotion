# G1 Humanoid Locomotion — State-Estimation-Aware RL (WIP)

> **Status: work in progress.** This is step one of a larger project applying
> recent state-estimation research to legged locomotion RL. Right now it's a
> proof of concept: swap the simulator's privileged velocity signal for a
> Kalman-filtered, sensor-derived one and confirm the G1 humanoid can still
> learn a working gait on it. Everything past that (a real EKF with bias
> estimation, closing the sim-to-real observation gap, quantified transfer
> results) is future work — see [Roadmap](#roadmap).

## What this is

A manager-based IsaacLab RL task for velocity-tracking locomotion on the
Unitree G1 humanoid, rough and flat terrain. Instead of training against the
simulator's ground-truth base velocity — an oracle signal no real robot has —
the policy is trained against an estimate built from IMU integration fused
with leg-odometry zero-velocity updates. The eventual goal is to use this as
a testbed for whether better state estimation (closer to what real hardware
would give you) changes what gaits RL discovers, and how well they'd survive
sim-to-real transfer.

## Current state (step 1)

`g1_observations.estimated_base_lin_vel` fuses:

- **Prediction** — accelerometer reading rotated to world frame (`quat_w`)
  plus gravity, integrated: `v += (R(q)·a_body + g) · dt`.
- **Correction** — leg odometry: for each foot currently in contact
  (`net_forces_w` above a force threshold), the base velocity implied by a
  zero-velocity foot constraint is computed from the leg Jacobian and joint
  velocities, then averaged across stance feet.
- **Fusion** — a per-axis (diagonal-covariance) Kalman filter combines the
  two; during flight phases (no foot in contact) the correction step is
  skipped and the prediction carries forward uncorrected.

This is mechanically a **linear per-axis KF**, not yet a true EKF — no
Jacobian linearization is needed since the transition/measurement models as
written are already linear. Two known gaps to close next:

- The accelerometer bias injected into the `imu_lin_acc` *observation*
  (`imu_lin_acc_biased`, reset via `reset_imu_bias`) isn't the signal the
  estimator integrates — `estimated_base_lin_vel` reads the clean
  `imu.data.lin_acc_b`. Same for contact: the noisy `foot_contact_booleans`
  observation and the filter's internal stance detection are computed
  independently. So the estimator never has to cope with the noise the
  policy itself sees.
- There's no attitude or bias state — just velocity.

Gait quality is shaped by ~12 reward terms (symmetry, swing/stance knee
extension, torso uprightness, double support, velocity-conditioned gait
reference, anti-hop terms) on top of the base `mdp` locomotion rewards.

### Milestones reached

- **Estimator validated in isolation** — leg-odometry Jacobian correctly
  reconciled against IsaacLab/PhysX API shape mismatches (floating-base DOF
  offset, per-body contact indexing); closed-loop XY velocity estimate
  converged to **0.26 m/s MAE** against ground truth during training.
- **RL converges on the estimated signal, not the oracle one** — reward
  **-8.6 → 23.3** over 820/1500 iterations, **99.3% episode-timeout rate**
  (i.e. the policy is surviving to the horizon, not falling), confirming the
  estimator is usable as a training signal, not just accurate offline.
- **Diagnosed and fixed a full class of gait pathologies**, not just
  patched symptoms: asymmetric leading-leg shuffle → gait/joint-symmetry
  rewards; half-swing on one leg → hip-pitch-opposition + velocity-scaled
  gait reference; single-leg hopping on rough terrain → traced to
  `feet_air_time` literally paying for flight phase, disabled, replaced with
  explicit flight-phase and base-height penalties.
- **Two independent external code reviews incorporated**, including a
  disagreement resolved on evidence (a reviewer flagged the policy as
  "unlikely to succeed" from static analysis alone; the 99.3% timeout rate
  said otherwise — kept the empirical result, fixed the gait-quality issues
  the review *did* get right).

## Results

*TODO — training curves, gait comparisons, and quantified metrics go here
once step 1 finishes training.*

**Training curves**

<!-- ![reward curve](docs/media/reward_curve.png) -->

**Gait — flat terrain**

<!-- ![flat terrain gait](docs/media/flat_gait.gif) -->

**Gait — rough terrain**

<!-- ![rough terrain gait](docs/media/rough_gait.gif) -->

**Video**

<!--
https://github.com/user-attachments/assets/REPLACE_ME
-->

## Roadmap

Phase 0 and Phase 1 are done; the rest is ordered so each phase produces a
result that's meaningful on its own, not just a prerequisite for the next.

### ✅ Phase 0 — Estimator implementation & validation
IMU-integration + leg-odometry Kalman filter implemented as a drop-in
replacement for privileged `base_lin_vel`; Jacobian/API fragility across
IsaacLab versions resolved; offline-validated against ground truth.

### ✅ Phase 1 — RL convergence on estimated state
PPO policy trained end-to-end against the KF estimate (not the oracle
signal) to convergence (reward 23.3, 99.3% timeout) on flat terrain, proving
the estimator is viable as a training-time observation, not just accurate
in isolation.

### 🔄 Phase 2 — Gait quality, in progress
Systematically eliminating gait pathologies (shuffle, half-swing, single-leg
hop on rough terrain) via reward-term diagnosis rather than blind reward
tuning — each fix tied to a specific mechanism (e.g. `feet_air_time` paying
for flight phase). Remaining before this phase closes:
- [ ] Wire the existing (currently dead) joint-order verification into every
      reward/observation function that indexes joints by name, closing a
      silent-failure risk before it contaminates a longer run
- [ ] Make `gait_reference_tracking` velocity-conditioned (frequency/amplitude
      scaled by commanded speed, not fixed at 1.0 Hz)
- [ ] Fix `feet_height_reward` to use terrain-relative height, not world-Z
      (currently breaks on rough terrain by construction)
- [ ] Collapse the duplicated reward-override block in
      `rough_env_cfg.__post_init__` into a single source of truth
- [ ] Restart training from scratch on the cleaned config and confirm the
      hop/shuffle pathologies don't reappear

### ⬜ Phase 3 — Rigorous comparison, not just convergence
- [ ] Multi-seed training runs (n ≥ 3) for statistical confidence, not
      single-run anecdote
- [ ] Controlled ablation: privileged velocity vs. current KF vs. Phase 4
      ESKF, reward terms held fixed, compared on tracking error, gait
      symmetry, and episode survival
- [ ] Formal gait-quality metrics beyond reward: stance/swing duty factor,
      left-right symmetry index, foot-clearance distribution — not just
      "does it look right in the video"

### ⬜ Phase 4 — Error-state EKF upgrade
- [ ] Extend the state from velocity-only to `[δv, δθ, b_a, b_g]`, with the
      leg-odometry residual as a genuinely nonlinear measurement function of
      attitude and bias (derivation already worked out, not yet implemented)
- [ ] Route the same corruption used for the policy's observations (IMU bias,
      contact-flip probability) into the estimator's own prediction/update
      steps, closing the current train/observe inconsistency where the filter
      never has to cope with the noise the policy sees

### ⬜ Phase 5 — Sim-to-real
- [ ] Domain-randomization audit specifically for estimator-relevant
      parameters (IMU noise density, foot-contact force threshold, joint
      encoder noise), not just visual/dynamics randomization
- [ ] Quantify estimator drift and velocity-tracking error under push and
      terrain-perturbation tests, in sim, as a stand-in for hardware
      robustness before deployment
- [ ] Deploy to physical G1 hardware; report estimator accuracy against a
      motion-capture or external ground-truth reference

### ⬜ Phase 6 — Writeup
- [ ] Consolidate Phase 3 ablation results, Phase 5 hardware numbers, and
      the gait-pathology diagnoses from Phase 2 into a workshop-paper-ready
      comparison (privileged vs. KF vs. ESKF, sim vs. real)

To install into an existing IsaacLab conda/venv:

```bash
pip install -e source/g1_locomotion
```
