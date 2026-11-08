"""Plot saved conservative predictions and validation histories without inference."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .io import read_json
from .statistics import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--function", type=int, default=0)
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--time", type=float)
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()
    archives = [np.load(p / "predictions.npz", allow_pickle=False) for p in args.evaluation]
    summaries = [read_json(p / "metrics.json") for p in args.evaluation]
    first = archives[0]
    for archive in archives[1:]:
        for key in ("ids", "truth", "times"):
            if not np.array_equal(archive[key], first[key]):
                raise ValueError("Comparison plots require identical references, IDs and times")
    index = len(first["times"]) - 1 if args.time is None else int(np.argmin(abs(first["times"] - args.time)))
    selection = (args.function, index, args.component)
    truth = first["truth"][selection]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if truth.ndim == 1:
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        x = (np.arange(len(truth)) + .5) / len(truth)
        axes[0].plot(x, truth, color="black", label="Reference")
        for archive, summary in zip(archives, summaries):
            value = archive["prediction"][selection]
            label = LABELS.get(summary["method"], summary["method"])
            axes[0].plot(x, value, label=label)
            axes[1].plot(x, np.abs(value - truth), label=label)
        axes[0].set(xlabel="x", ylabel="Conservative component", title=f"t={first['times'][index]:g}")
        axes[1].set(xlabel="x", ylabel="Absolute physical error", ylim=(0, None))
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axes[0].legend(fontsize=8)
    else:
        values = [truth] + [a["prediction"][selection] for a in archives]
        low, high = min(v.min() for v in values), max(v.max() for v in values)
        figure, axes = plt.subplots(1, len(values), figsize=(3.2 * len(values), 3.4), constrained_layout=True, squeeze=False)
        for axis, value, label in zip(axes[0], values, ["Reference"] + [LABELS.get(s["method"], s["method"]) for s in summaries]):
            displayed = axis.imshow(value.T, origin="lower", extent=(0, 1, 0, 1), vmin=low, vmax=high)
            axis.set(title=label, xlabel="x", ylabel="y")
        figure.colorbar(displayed, ax=axes.ravel().tolist(), shrink=.8)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    for archive in archives:
        archive.close()
    if args.history:
        history = read_json(args.history)
        figure, axis = plt.subplots(figsize=(5, 3.3), constrained_layout=True)
        axis.semilogy([r["update"] for r in history], [r["loss"] for r in history], marker="o")
        axis.set(xlabel="Optimizer update", ylabel="Validation MSE")
        figure.savefig(args.output.with_name(args.output.stem + "-validation" + args.output.suffix), dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
