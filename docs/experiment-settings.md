# Experiment settings

This document records settings behind the current two-branch experiments.
Production configurations are `configs/{swe,euler,swe2d}.json`; configurations
whose names begin with `smoke-` deliberately change grid sizes, function counts,
sequence lengths and update budgets for execution checks only.

## Systems, arrays and fixed scaling

The domain is periodic with unit side length. 1D shallow water uses conservative
components `(h, hu)` and gravity one. Euler uses `(rho, rho*u, E)` and gamma 1.4.
2D shallow water uses `(h, hu, hv)`. Neural states have shape `(B,C,N)` or
`(B,3,N,N)`; trajectories add a time dimension after the batch dimension. The
1D reference solver uses `(B,N,C)` internally.

The native neural time step is 0.02. Training arithmetic is float32 without AMP
or TF32; references are integrated in float64 and conservatively restricted.
Runtime train/validation state arrays are converted to float32, with function
identities and labels preserved. Fixed channel scales are `(1,1)` for SWE,
`(1,1,2.5)` for Euler and identity for 2D SWE; no held-out normalization is fitted.

## Architecture and two-branch initialization

Each current D-CINO hidden block contains a Fourier map plus an unrestricted
signed local stencil followed by GELU. 1D uses width 48, 16 retained rFFT modes,
three hidden blocks and five-tap periodic stencils. 2D uses width 24, four modes
in each of two rFFT slabs, three hidden blocks and circular 5-by-5 stencils.
The 1D bins are 0 through 15. The two 2D slabs use x bins 0,1,2,3 and
-4,-3,-2,-1, with y bins 0,1,2,3 in both.
The lift is a pointwise affine map. The output head has two pointwise affine
maps with a GELU between them. The residual coefficient is 0.2. The dyadic map
uses one shared network at the full state and the half perturbation around its
conservative mean, and forms `2*(Phi(U)-2*Phi((U+mean(U))/2))`. Mean subtraction
is applied after the increment.

To retain the historical zero-update distribution, an auxiliary constructor
draws the original affine and local weights, adds the affine matrix to the
central local tap and adds the biases, then deletes the independent affine
branch. Optimization starts only after this folding. The resulting trainable
model has two branches; no old optimizer moments are transferred. Floating-point
association can differ from the pre-folding representation. Original supplied
initialization payloads are preferred when matching a historical run exactly.
The package also supports direct seeded construction in the installed PyTorch
version and does not promise identical RNG tensors across software versions.

The analytic carrier linearizes at the conserved mean. 1D separates acoustic
and, for Euler, contact characteristics. 2D separates longitudinal acoustic and
transverse shear modes jointly in Fourier space. The prescribed spectral
viscosity has alpha 0.05, power 2 and cutoff `ceil(sqrt(N))`. The same real scalar
multiplier acts on every conservative component. Real projection on even-grid
Nyquist modes is retained; exact group statements need the stated Nyquist
restriction. `no_sv` sets alpha to zero. `no_dyadic` replaces the dyadic difference
by one full-state network evaluation and leaves all other operations unchanged.

The radial correction first restores the input conservative mean. It then
contracts toward that mean until the density/depth floor 1e-7 and the inherited
total-entropy tolerance hold. Euler also checks internal energy. It uses 24
bisections and tolerance `8*eps*(1+abs(target_entropy))`. Candidate, corrected
state and reconstructed observation have distinct roles. No manual scheduled
parameter overwrite is used.

Native 1D FNO uses width 44, 16 Fourier modes, four Fourier/pointwise blocks and
a width-88 pointwise projection head. Native 2D FNO uses width 24, four modes,
four blocks and a width-48 head. GELU follows every block except the last.
The 1D periodic U-Net uses base width 21, three average-pooling levels, nearest
upsampling with pointwise reduction, double circular three-tap convolutions and
ReLU. The 2D U-Net uses base width 9, three max-pooling levels and learned
transposed-convolution upsampling. Native baselines predict the complete next
state without a carrier, mean projection or radial correction. Residual FNO
uses `U + dt*P0*N(U)`, the same fixed scales and radial correction as D-CINO.
Carrier-only uses exactly the SV carrier and correction with no learned tensors.

## Training and validation selection

All learned comparisons use seeds 1621, 241 and 461. AdamW uses learning rate
0.001, weight decay 1e-5 and global gradient-norm clipping at one. A batch has
16 functions, sampled with replacement as eight functions from each family.
Sampling uses `numpy.default_rng(26101550 + seed)`: an `(updates,2,8)` array of
uniform ranks selects indices within the original ordered family arrays. The
sequence endpoints cycle through 4, 8 and 16 native steps, corresponding to
windows 0.08, 0.16 and 0.32. Every window starts from the true initial state;
all later inputs are the model's own corrected predictions. The loss averages
fixed-scale conservative MSE across every native step, batch, channel and cell.

Validation uses every native time through 0.8, with equal function/time weights
and batches of at most eight functions. Checkpoint selection includes update
zero and every 100 updates: the strictly smallest raw validation loss selects
`best.pt`, and exact ties retain the earlier state. The separate improvement
anchor requires a 0.2 percent relative decrease; otherwise its stale counter
increases. ReduceLROnPlateau uses factor 0.5, patience four validation checks,
relative threshold 0.002 and minimum learning rate 1e-5. Stopping requires at
least 1000 updates and 12 stale checks, with a maximum of 6000 updates and an
operational 7200-second training ceiling checked before an update. Dataset,
model and optimizer setup are outside that timing boundary; initial validation
and checkpoint I/O are inside it. Budget completion does not certify convergence.

Training never reads the test split. Pauses preserve model, optimizer, scheduler,
RNG, sampling position and the selected checkpoint identity. Successful and
failed terminals cannot be silently reopened. Numerical failures are retained;
missing fitted seeds make three-fit statistics incomplete.

## Reference generation and function isolation

Both solvers use primitive-variable MUSCL-HLL reconstruction with monotonized
central slopes, SSPRK2, periodic boundaries and CFL 0.35. The 2D divergence is
unsplit and its CFL combines the x/y rates. Candidate stages are tested for
admissibility; at most 16 step-halving trials are made. Solvers land on each
requested output time. Their shared time step depends on the complete solver
batch, so the recipe's batch boundaries are scientifically relevant.

1D training starts with 96 functions at seed 104729 (Euler adds 7919), with
48 smooth and 48 discontinuous functions. It appends the odd indices of 1952
draws at seed 120104729 (Euler adds 7919), giving 976 additional discontinuous
functions and 1072 functions in total. The original 96 are integrated together
through 0.1; the extras use batches of 128 through 0.1. From their stored t=0.1
states, batches of 32 are continued for another 35 intervals of 0.02. The two
segments are concatenated and restricted from 512 cells to 128 cells. The
segmented construction is preserved rather than replaced by a single solve.

1D validation uses 24 fresh functions at seed 26101531 (Euler adds 7919),
12 from each family, integrated together on 512 cells at every 0.02 through
0.8, then restricted to 128. General test uses seed 26101591 (Euler adds 7919),
32 fresh functions integrated together on 512 cells, with stored times every
0.08 through 0.8. The neural forecast still advances at every 0.02. The original
1D refinement check repeats the first four 512-cell piecewise-constant initial
states onto 1024 cells; independent midpoint redrawing is a different check.

2D parameter generation alternates nonparallel smooth waves and circular dams.
Training has 128 functions at seed 9281101, validation 12 at seed 26101521 and
general test 32 at seed 26101581. Primitive fields are sampled at master512
midpoints, converted to conservative states and shifted to zero mass-weighted
mean velocity there; block averaging produces the reference256 grid. Training
integrates in batches of four to 0.4 (21 states), validation and general test in
batches of two to 0.8 (41 states). All are restricted to 64-by-64 observations.
The same parameter identities are shared across resolution and amplitude
variants; variants are not independent functions.

The 1D weak generator draws 80 functions on 1024 cells at seed 120181081
(Euler adds 7919), selects even indices 0,2,...,78, and rescales the same
mean-centered conservative directions at amplitudes 1/16 and 1/64. Each
40-function batch is integrated together at times 0.02,...,0.2 and restricted
to 128 cells. The 2D weak generator uses 64 functions at seed 10053303,
master512 quadrature, reference256, primitive amplitude 1/16 and batches of
four functions at the same ten output times. This is the EXP-012 / EXP-2D-003
weak construction, distinct from the current general-test populations. The
historical 2D first-eight-function refinement has four smooth and four dam
functions and is not a 32-function general-reference bound.

The built-in generator covers these weak datasets as well as training,
validation and general test. Existing arrays can be supplied through
`data-map.json`. Boosted/axial/diagonal collections and historical refinement
tables remain retained analyses. Source hashes and exact historical asset
identities are retained by the authors. Full regeneration has not been rerun
during extraction of this package.

## Forecasts, metrics and statistics

D-CINO, carrier-only and residual FNO remove initial mass-weighted mean velocity,
retain a canonical trajectory and reconstruct physical observations only when
requested. The observations do not feed back. The 1D remap is conservative
linear periodic interpolation; 2D uses the sequential bilinear periodic remap.
The physical arm advances the same corrected map directly in the physical frame.
Native FNO/U-Net use physical autoregression.

Metrics retain physical mean bias. Centered shape error and mean error are
additional diagnostics. Weak errors divide physical state RMS by the endpoint
truth fluctuation RMS, with a denominator floor 1e-14. Numerical failure uses
the inherited absorbing mask and penalty 1e6 for all affected reported endpoints;
no failed function is omitted from the mean. The historical 1D corrected D-CINO
path treats finite physical inadmissibility as a forecast failure, in both
forecast frames, and stores the last valid observation with its cause/mask.
This corrected 1D path has no separate absolute-state magnitude-cap check.
The 1D native/residual baseline path records finite physical invalidity
separately and retains its actual physical error; its numerical trigger is
nonfinite state or `max(abs(U/S)) > 1e6`. The 2D path retains finite negative-depth accuracy and
flags invalidity separately; nonfinite or absolute states exceeding 1e6 trigger
the numerical failure mask. These method-specific inherited conventions are
preserved rather than replaced by one new common clipping rule.
The unrepaired 1D raw-carrier diagnostic is a fourth path: it rejects nonfinite
candidate/observed states only, with neither a physical-floor rejection nor a
separate magnitude trigger. Its finite inadmissible values remain diagnostic
outputs and do not replace corrected carrier-only predictions.

For each fixed fitted seed, errors are averaged arithmetically over functions;
the three fitted means are reported as mean and sample standard deviation.
Function uncertainty uses 10000 paired family-stratified resamples. The three
fixed-seed errors are averaged per function before resampling, so cells, output
times, amplitudes, boosts and seeds are not treated as independent replicates.
The stable resampling seed is 26101699 plus SHA256-derived population/family
words. Ratios use arithmetic function means with common draws; any zero
denominator bootstrap replicate leaves the ratio interval undefined. Carrier
controls have no training seed or training-seed standard deviation.

For the common-carrier response measurement, seed 1621 and the first smooth
weak function at amplitude 1/16 are fixed. Remove its mass-weighted mean
velocity, set A to its conservative mean and z to its centered state divided by
1/16, without RMS renormalization. Eleven logarithmically spaced amplitudes
from 1e-6 to 1e-1 retain this same conservative direction. Real and complex
checkpoint tensors are explicitly promoted to float64/complex128 and every
tensor is checked for exact equality after conversion back to its original
dtype. The checkpoint file is unchanged. The response subtracts the constant
step and the common SV carrier; the exported arrays also retain pre-correction,
undamped-carrier and constant-bias diagnostics. This response is separate from
physical PDE prediction error. `dcino.response` and `dcino.plot_response` provide
the executable measurement and trend figure.

Current isolated latency measurement uses batch one and the first smooth
general-test function. One corrected model step with its finite-state check
and the complete 40-step forecast are timed separately. Every instance receives
three warmups per arm and 20 synchronized repetitions. Even measurement blocks
visit the fixed cohort order; odd blocks reverse it. Inputs are device-resident,
and timing includes actual frame transforms, correction, output storage and
dispatcher diagnostics while excluding metric reduction, data loading, I/O,
reference generation and transfers. No concurrent training or other device
inference should run. Each fit retains all repeats, median and interquartile
range. Full forecast cost is not inferred by multiplying one-step cost by 40.
Hardware, thread count and software version accompany newly measured values;
changing them does not reproduce a historical latency entry.

The reported full/ablation fits used local PyTorch 2.11, two CPU threads in 1D
and an NVIDIA GeForce RTX 3050 Ti Laptop GPU in 2D. Retained native baselines
include six preceding-study 1D seed-1621 fits, twelve further local-GPU 1D fits
and nine cloud-GPU 2D fits. Native/residual FNO share their initial random
tensors at each seed. Common optimization rules therefore do not imply equal
compute. The retained study has 54 normally completed fits plus deterministic
carrier controls; individual selected updates, unsuccessful attempts and
hardware/load records remain with their historical evidence rather than being
new trials in this extraction. Of the 36 primary fits, 27 reached 6000 updates
and nine stopped on the validation plateau. The 18 new ablation fits completed
96800 updates in total: 14 reached the limit and four stopped on the plateau.
These are retained study outcomes, not training performed for this package.

## Extraction verification

Only short CPU execution checks and retained-source numerical equivalence were
run for this package. Full 6000-update fits, full reference regeneration, GPU
throughput, all weak directions, frame sweeps and published confidence intervals
were not recomputed. No new smoke output is evidence for a paper performance
claim. Consult the authors' retained evidence for the exact data/checkpoint
bindings behind a historical figure or table.
