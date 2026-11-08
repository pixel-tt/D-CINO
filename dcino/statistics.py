"""Current fixed-fit, stratified paired-function statistics (no inference)."""
from collections import defaultdict
import json
import hashlib
import numpy as np
from pathlib import Path
from .io import read_json, write_json, sha256


def verified_path(item):
    path = Path(item["path"])
    if item.get("sha256") and sha256(path) != item["sha256"]:
        raise ValueError("Metric archive checksum mismatch")
    return path

SEEDS = (1621, 241, 461)
DETERMINISTIC = {"carrier_only", "raw_carrier"}
METHODS = ("dcino", "carrier_only", "fno", "unet", "fno_physical_wrapper",
           "no_sv", "no_dyadic", "raw_carrier")
MAIN = METHODS[:5]
CONTROLS = ("dcino", "carrier_only", "no_sv", "no_dyadic", "raw_carrier")
LABELS = {"dcino": "D-CINO", "carrier_only": "carrier-only", "fno": "FNO",
          "unet": "U-Net", "fno_physical_wrapper": "Residual FNO",
          "no_sv": "No SV", "no_dyadic": "No dyadic", "raw_carrier": "Raw carrier (diagnostic)",
          "reference": "Reference"}
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#E69F00", "#009E73",
                           "#8E44AD", "#CC79A7", "#56B4E9", "#666666")))
COLORS["reference"] = "#222222"
STYLES = dict(zip(METHODS, ("-", "--", "--", "-.", ":", "-.", ":", "--")))
STYLES["reference"] = "-"
MARKERS = dict(zip(SEEDS, ("o", "s", "^")))

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def frame_key(value):
    return tuple(float(v) for v in (value if isinstance(value, list) else [value]))


def panel_key(row):
    return (row["system"], row["dataset_id"], frame_key(row["frame"]),
            float(row["endpoint"]), row["metric"], row.get("view", "published_dispatch"))


def metric_rows(packet):
    rows = packet.get("metrics", [])
    if "metrics_file" in packet:
        path = verified_path(packet["metrics_file"])
        if path.suffix.lower() == ".jsonl":
            raw = path.read_bytes()
            rows = [json.loads(line) for line in raw.decode("utf-8-sig", errors="strict").splitlines() if line.strip()]
        else:
            value = read_json(path)
            rows = value["rows"] if isinstance(value, dict) else value
    required = {"system", "dataset_id", "bootstrap_population_id", "family", "function_id",
                "base_function_id", "method", "seed", "fit_id", "frame", "endpoint", "metric",
                "value", "numerical_failure", "physical_invalid", "denominator_floor"}
    identities = set()
    fits = {}
    for row in rows:
        if not required.issubset(row):
            raise ValueError("Missing metric fields: " + str(required - set(row)))
        if row["method"] not in METHODS:
            raise ValueError("Unknown method: " + row["method"])
        deterministic = row["method"] in DETERMINISTIC
        if (deterministic and row["seed"] is not None) or (not deterministic and row["seed"] not in SEEDS):
            raise ValueError("Seed-free and trained-method identities must remain distinct")
        if not np.isfinite(float(row["value"])):
            raise ValueError("Metrics must retain a finite evaluator-defined failure penalty")
        identity = panel_key(row) + (row["method"], row["seed"], str(row["function_id"]))
        if identity in identities:
            raise ValueError("Duplicate per-function metric row: " + str(identity))
        identities.add(identity)
        fit_key = (row["system"], row["method"], row["seed"])
        if fits.setdefault(fit_key, row["fit_id"]) != row["fit_id"]:
            raise ValueError("A fixed trained fit changed between evaluation panels")
    return rows


def bootstrap_indices(system, population, family, count, replicates=10000, seed=26101699):
    """One stable stream per base population/family, reused for all fixed fits.

    SHA256-derived seed words avoid Python's randomized hash and dependence on
    panel traversal order. Original family sample counts remain unchanged.
    """
    token = canonical([system, population, family]).encode("utf-8")
    words = np.frombuffer(hashlib.sha256(token).digest()[:16], dtype="<u4").tolist()
    rng = np.random.default_rng(np.random.SeedSequence([seed, *words]))
    return rng.integers(0, count, size=(replicates, count))


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def summarize(rows, replicates=10000, expected_methods=None):
    """Reduce exact evaluator metrics and retain all incomplete/failure outcomes."""
    panels = defaultdict(list)
    for row in rows:
        panels[panel_key(row)].append(row)
    summaries, contrasts = [], []
    populations = {}
    for key, panel in sorted(panels.items(), key=lambda item: canonical(item[0])):
        system, dataset, frame, endpoint, metric, view = key
        family_names = sorted({row["family"] for row in panel})
        population_ids = {row["bootstrap_population_id"] for row in panel}
        if len(population_ids) != 1:
            raise ValueError("A panel must bind one base-function population")
        population = next(iter(population_ids))
        for family in ["all", *family_names]:
            selected = panel if family == "all" else [r for r in panel if r["family"] == family]
            functions = {}
            for row in selected:
                ident = str(row["base_function_id"])
                info = (str(row["function_id"]), row["family"])
                if ident in functions and functions[ident] != info:
                    raise ValueError("One base identity maps to conflicting functions in a panel")
                functions[ident] = info
            ids = sorted(functions, key=lambda ident: (functions[ident][1], ident))
            by_family = {name: [ident for ident in ids if functions[ident][1] == name]
                         for name in sorted({v[1] for v in functions.values()})}
            draws = []
            for name, family_ids in by_family.items():
                roster_key = (system, population, name)
                prior = populations.setdefault(roster_key, family_ids)
                if prior != family_ids:
                    raise ValueError("Coupled bootstrap population changed base-function identities")
                local = bootstrap_indices(system, population, name, len(family_ids), replicates)
                positions = np.asarray([ids.index(ident) for ident in family_ids])
                draws.append(positions[local])
            indices = np.concatenate(draws, axis=1)
            common = {"system": system, "dataset_id": dataset, "frame": list(frame), "endpoint": endpoint,
                      "metric": metric, "family": family, "view": view, "bootstrap_population_id": population,
                      "function_count": len(ids)}
            method_values = {}
            planned_methods = (expected_methods.get(view, []) if isinstance(expected_methods, dict)
                               else expected_methods) or [m for m in METHODS if any(r["method"] == m for r in selected)]
            for method in planned_methods:
                method_rows = [r for r in selected if r["method"] == method]
                expected = (None,) if method in DETERMINISTIC else SEEDS
                cells = {(r["seed"], str(r["base_function_id"])): r for r in method_rows}
                missing = [[seed, ident] for seed in expected for ident in ids if (seed, ident) not in cells]
                record = dict(common, method=method, seed_order=list(expected), complete=not missing,
                              missing_seed_function_pairs=missing, mean=None, sample_sd=None,
                              per_seed_mean={}, mean_ci_95=None,
                              numerical_failure_rows=sum(bool(r["numerical_failure"]) for r in method_rows),
                              physical_invalid_rows=sum(bool(r["physical_invalid"]) for r in method_rows),
                              denominator_floor_rows=sum(bool(r["denominator_floor"]) for r in method_rows))
                if not missing:
                    matrix = np.asarray([[float(cells[(seed, ident)]["value"]) for ident in ids] for seed in expected])
                    seed_means = matrix.mean(axis=1)
                    record["per_seed_mean"] = {str(seed): float(v) for seed, v in zip(expected, seed_means)}
                    record["mean"] = float(seed_means.mean())
                    record["sample_sd"] = None if method in DETERMINISTIC else float(seed_means.std(ddof=1))
                    per_function = matrix.mean(axis=0)
                    boot = per_function[indices].mean(axis=1)
                    record["mean_ci_95"] = interval(boot)
                    method_values[method] = (per_function, boot)
                summaries.append(record)
            if "dcino" not in method_values:
                continue
            denominator, denominator_boot = method_values["dcino"]
            for method, (numerator, numerator_boot) in method_values.items():
                if method == "dcino":
                    continue
                valid = denominator_boot != 0
                ratio = float(numerator.mean() / denominator.mean()) if denominator.mean() != 0 else None
                contrasts.append(dict(common, numerator=method, denominator="dcino",
                    difference=float((numerator - denominator).mean()),
                    difference_ci_95=interval(numerator_boot - denominator_boot),
                    ratio=ratio, ratio_ci_95=interval(numerator_boot / denominator_boot) if valid.all() else None,
                    zero_denominator_replicates=int((~valid).sum()), ratio_ci_defined=bool(valid.all())))
    return summaries, contrasts


def rows_from_evaluation(folder):
    """Adapt exported scalar metric arrays to the current per-function contract."""
    packet = read_json(folder / "metrics.json")
    times = packet["data"]["times"]
    values = packet["metrics"]
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        native_failed = archive["failure_mask"]
        native_times = archive["native_times"]
        match = [int(np.argmin(abs(native_times - time))) for time in times]
        failed = native_failed[:, match]
    invalid = np.asarray(values.get("physical_invalid_mask", np.zeros_like(failed)), dtype=bool)
    if invalid.shape != failed.shape:
        invalid = invalid[:, match]
    rows = []
    for metric, array in values.items():
        if isinstance(array, dict) or metric.endswith("mask"):
            continue
        array = np.asarray(array)
        if array.shape != failed.shape:
            continue
        for function, ident in enumerate(packet["function_ids"]):
            for step in range(1, len(times)):
                rows.append({"system": packet["problem"], "dataset_id": packet["dataset_id"],
                    "bootstrap_population_id": packet["bootstrap_population_id"],
                    "family": packet["families"][function], "function_id": ident, "base_function_id": ident,
                    "method": packet["method"], "seed": packet["seed"], "fit_id": packet["fit_id"],
                    "frame": packet["frame_velocity"], "view": packet["view"], "endpoint": times[step],
                    "metric": metric, "value": float(array[function, step]),
                    "numerical_failure": bool(failed[function, step]), "physical_invalid": bool(invalid[function, step]),
                    "denominator_floor": bool(np.asarray(values.get("reference_floor_active_mask",
                        values.get("denominators", {}).get("endpoint_denominator_floor_applied", np.zeros_like(failed))))[function, step])})
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluation", type=Path, nargs="+")
    group.add_argument("--rows", type=Path, help="Current normalized per-function row JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()
    rows = ([row for folder in args.evaluation for row in rows_from_evaluation(folder)]
            if args.evaluation else read_json(args.rows))
    rows = metric_rows({"metrics": rows})
    summaries, contrasts = summarize(rows, args.replicates)
    write_json(args.output, {"summaries": summaries, "contrasts": contrasts,
        "scope": "function uncertainty conditional on three fixed fits; incomplete fits retain null summaries"})


if __name__ == "__main__":
    main()
