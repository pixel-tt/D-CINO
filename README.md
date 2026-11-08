# D-CINO reproducibility core

This package contains the current two-branch D-CINO implementation for periodic
1D shallow water, 1D Euler and 2D shallow water. It includes the analytic spectral
carrier, spectral viscosity, dyadic interaction, mean projection, radial
correction, deferred output reconstruction, training, reference generation and
the principal comparison methods. No private workspace is needed.

The native FNO and periodic U-Net predict the next conservative state directly.
Residual FNO uses the prescribed mean-free residual and radial correction.
Carrier-only has zero trainable parameters. `no_sv` and `no_dyadic` each change
one operation in D-CINO. `raw_carrier` is an unrepaired diagnostic.

## Install and run a small example

Use Python 3.10 or later in an environment with PyTorch, NumPy and Matplotlib:

```sh
python -m pip install -r requirements.txt
python -m dcino.data --config configs/smoke-swe.json --split train validation --output results/smoke-swe/data
python -m dcino.train --config configs/smoke-swe.json --data results/smoke-swe/data --output results/smoke-swe/run
python -m dcino.data --config configs/smoke-swe.json --split test --output results/smoke-swe/data
python -m dcino.evaluate --config configs/smoke-swe.json --data results/smoke-swe/data --run results/smoke-swe/run --output results/smoke-swe/evaluation
python -m dcino.plot --evaluation results/smoke-swe/evaluation --output results/smoke-swe/profile.png --history results/smoke-swe/run/history.json
```

Replace `smoke-swe` by `smoke-euler` or `smoke-swe2d` for the other systems.
These profiles use small grids and two optimizer updates. They check execution;
they are not retrained paper results or a meaningful performance comparison.
All commands provide `--help`.

## Scientific runs

`configs/swe.json`, `configs/euler.json` and `configs/swe2d.json` retain the paper
training settings and reference recipes. Full reference generation and training
are expensive. The 1D training recipe preserves the original short trajectory
and its batched continuation; the 2D recipe preserves master-grid quadrature and
the original solver batches. Details are in
[docs/experiment-settings.md](docs/experiment-settings.md).

```sh
python -m dcino.data --config configs/swe.json --split train validation --output data/swe
python -m dcino.train --config configs/swe.json --data data/swe --method dcino --seed 1621 --output results/swe/dcino-1621
```

Run each learned method with all seeds `1621`, `241`, `461`, then generate/open
the test split and evaluate the validation-selected `best.pt`. The training
entry point reads only train and validation. Test data must never select
architecture, hyperparameters, stopping, seed or checkpoint. Terminal runs are
preserved; no automatic retry drops failures. Use separate output directories
for each run. `--pause-after` or Ctrl-C saves a nonterminal state; `--resume`
restores model, optimizer, scheduler, RNG and sampling cursor.

Use `--method fno`, `unet`, `fno_physical_wrapper`, `no_sv` or `no_dyadic` for the
learned comparisons. Evaluate the parameter-free control without a run:

```sh
python -m dcino.evaluate --config configs/swe.json --data data/swe --method carrier_only --output results/swe/carrier
```

`--frame physical` on D-CINO or carrier-only changes the forecast frame while
keeping the single-step map. The default uses the published method-specific
frame. Saved failures, physical invalidity and physical errors remain separate;
the original numerical penalty is retained.

To use existing archives, provide `data/data-map.json` describing relative
paths, array names, layouts and optional SHA256 values. See
[configs/data-map.example.json](configs/data-map.example.json). Large original
arrays and checkpoint assets are not bundled. `--initialization FILE` accepts
an original zero-update checkpoint for exact starting tensors; folding occurs
before a fresh AdamW optimizer. A fresh seed on another PyTorch version does
not guarantee historical tensor identity.

## Outputs and comparison statistics

Evaluation saves per-function metric arrays, failure masks and conservative
fields. `prediction`, `truth` and scalar metrics use the stored reference
`times`; `theta`, `failure_mask` and the 1D `physical_invalid_mask_native` retain
the `native_times` axis. This distinguishes sparse 1D reference outputs from
the intervening neural steps. `dcino.plot` accepts several evaluation directories for a shared-reference
comparison. `dcino.statistics --evaluation DIR ... --output summary.json`
uses the current family-stratified paired-function bootstrap. Three fitted seeds
are held fixed, averaged before resampling functions, and separately reported
with sample standard deviation. Carrier controls are seed-free. Incomplete
three-fit panels retain null summaries; a smoke run therefore has no three-seed
confidence interval. The same module accepts the normalized per-function row
contract through `--rows FILE` for retained paper metric exports.

`dcino.benchmark` measures single steps and complete 40-step forecast calls
separately after three warmups, retaining 20 raw repeats in alternating/reversed
cohort order. It includes radial correction and observation reconstruction and
excludes reference generation and file I/O. Its result depends on the local
device, software and thread count. Run it without concurrent device workloads.

## Weak waves and response measurements

The generator also provides both 1D conservative-direction amplitudes (`weak`
at 1/16 and `veryweak` at 1/64) and the 2D primitive-amplitude weak family. These
use the original independent function recipes and solver batch boundaries.
For a completed seed-1621 run, the response diagnostic uses the first smooth
weak function, its fixed conservative direction, and exact promotion of saved
real/complex parameters with conversion-back equality checks:

```sh
python -m dcino.data --config configs/swe.json --split weak veryweak --output data/swe
python -m dcino.evaluate --config configs/swe.json --data data/swe --run results/swe/dcino-1621 --split weak --output results/swe/weak
python -m dcino.response --config configs/swe.json --data data/swe --run results/swe/dcino-1621 --output results/swe/response
python -m dcino.response --config configs/swe.json --data data/swe --method carrier_only --output results/swe/carrier-response
python -m dcino.plot_response --response results/swe/response results/swe/carrier-response --output results/swe/response.png
```

Use only `--split weak` in the 2D generator. These commands cover general and
weak physical accuracy, 1D solution profiles, 2D solution images and the main
common-carrier response trend. Bulk-velocity grids, inverse canonical scoring,
historical reference-refinement tables and exact manuscript panel composition
remain separate retained analyses. Newly generated tables and timing results
need their own data/checkpoint/software bindings.

## Code map and verified scope

- `dcino/models/factory.py`: current method constructors and carrier-only.
- `dcino/models/fusion.py`, `variants.py`: two-branch initialization and ablations.
- `dcino/models/one_dimensional/`: Fourier/stencil models, characteristic
  carrier, radial correction, NumPy reference solver and original scoring.
- `dcino/models/two_dimensional/`: unsplit acoustic/shear carrier, radial
  viscosity, reference solver, baseline models and original scoring.
- `dcino/data.py`, `data_math.py`: split recipes and master-grid quadrature.
- `dcino/train.py`, `training_math.py`: free autoregression and validation selection.
- `dcino/evaluate.py`, `response.py`, `statistics.py`, `benchmark.py`,
  `plot.py`, `plot_response.py`: scientific outputs.

The package was checked with compilation, CLI help, small CPU end-to-end runs
and numerical comparisons against the retained source implementation. Full
paper-budget training and full reference regeneration were not rerun for this
code extraction. Recreating a table also requires its exact reference arrays,
selected checkpoints and per-function data bindings; a newly generated example
does not reproduce a historical table. Source bindings remain with the authors.

No new software license, author list, DOI or public release URL is assigned here.
