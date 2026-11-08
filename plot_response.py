"""Plot saved common-carrier response discrepancies across the fixed amplitudes."""
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
    parser.add_argument("--response", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure, axis = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    identity = None
    for folder in args.response:
        packet = read_json(folder / "response.json")
        current = (packet["system"], packet["function_id"], packet["data"]["sha256"])
        if identity is not None and current != identity:
            raise ValueError("Response comparisons require the same physical input direction")
        identity = current
        rows = packet["values"]
        epsilon = np.asarray([r["epsilon"] for r in rows])
        full = np.asarray([r["full_step_Bnu_remainder_rms"] for r in rows])
        axis.loglog(epsilon, np.maximum(full, np.finfo(float).tiny), marker="o", label=LABELS.get(packet["method"], packet["method"]))
        if packet["method"] == "dcino":
            axis.loglog(epsilon, np.maximum([r["preQ_Bnu_remainder_rms"] for r in rows], np.finfo(float).tiny), linestyle=":", label="D-CINO before Q")
    axis.set(xlabel="Perturbation amplitude", ylabel="Common-carrier response discrepancy")
    axis.legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
