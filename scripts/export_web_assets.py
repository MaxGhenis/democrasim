"""Export compressed margin distributions and test vectors for the web essay.

The interactive essay on maxghenis.com computes tracking probabilities in
the browser using the same closed form as :func:`analytic_plurality_curve`.
This script exports everything that page needs:

- per-world compressed margin distributions (exact atoms below $100 — the
  levy staircase must survive — cents-rounded and $1-rounded aggregation
  above, then tail thinning), small enough to ship as JSON;
- toy-world parameters (Gaussian margin moments; homogeneous margin);
- headline constants for captions;
- test vectors computed with the *full* 120,408-row electorate, so the
  site's TypeScript implementation is pinned against this package's math
  AND against the compression error in one tolerance.

Run: uv run python scripts/export_web_assets.py [out.json]
The output is copied byte-for-byte into maxghenis.com/src/data/ (see that
repo's AGENTS.md); regenerate here, never hand-edit there.
"""

import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from democrasim import (
    Isoelastic,
    analytic_plurality_curve,
    apply_financing,
    load_measured_electorate,
)
from democrasim.data import ARTIFACT_STEM, _data_dir
from democrasim.electorate import Electorate

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "web" / "assets.json"

#: Margins with |m| below this are kept at cent precision (the levy
#: staircase lives here); larger margins aggregate to whole dollars.
ATOM_PRECISION_BOUND = 100.0
#: Aggregated pairs beyond this count are thinned by weighted quantiles.
MAX_PAIRS = 4_000


def compress_margins(electorate: Electorate) -> dict[str, list[float]]:
    """Aggregate (margin, weight) pairs small enough for the browser."""
    margins = electorate.margins
    weights = electorate.weights
    rounded = np.where(
        np.abs(margins) < ATOM_PRECISION_BOUND,
        np.round(margins, 2),
        np.round(margins, 0),
    )
    unique, inverse = np.unique(rounded, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)

    if len(unique) > MAX_PAIRS:
        # Keep the heaviest atoms exact; thin the rest by weighted quantiles.
        order = np.argsort(mass)[::-1]
        keep = order[: MAX_PAIRS // 2]
        rest = order[MAX_PAIRS // 2 :]
        rest_sorted = rest[np.argsort(unique[rest])]
        rest_margins = unique[rest_sorted]
        rest_mass = mass[rest_sorted]
        cum = np.cumsum(rest_mass)
        edges = np.searchsorted(cum, np.linspace(0, cum[-1], MAX_PAIRS // 2 + 1)[1:-1])
        groups = np.split(np.arange(len(rest_margins)), np.unique(edges))
        thin_margins, thin_mass = [], []
        for group in groups:
            if len(group) == 0:
                continue
            group_mass = rest_mass[group].sum()
            thin_margins.append(
                float(np.average(rest_margins[group], weights=rest_mass[group]))
            )
            thin_mass.append(float(group_mass))
        unique = np.concatenate([unique[keep], np.array(thin_margins)])
        mass = np.concatenate([mass[keep], np.array(thin_mass)])
        order = np.argsort(unique)
        unique, mass = unique[order], mass[order]

    return {
        "margins": [round(float(m), 2) for m in unique],
        "weights": [round(float(w), 3) for w in mass],
    }


def build_worlds(gross: Electorate) -> dict[str, Electorate]:
    totals = gross.household_dollars()
    sign_flip = gross.deltas.copy()
    sign_flip[:, 1] *= (totals[0] / totals[1]) * 1.03
    return {
        "per_capita": apply_financing(gross, "per_capita"),
        "proportional": apply_financing(gross, "proportional"),
        "none": gross,
        "sign_flip": apply_financing(
            gross.with_deltas(sign_flip, note="COUNTERFACTUAL B costs 3% more than A"),
            "per_capita",
        ),
    }


def test_vectors(worlds: dict[str, Electorate]) -> list[dict]:
    """Ground truth from the full electorate for the site's vitest suite."""
    vectors = []
    cases = [
        # (world, sigma, bias_toward_b, n_voters)
        ("per_capita", 0.0, 0.0, None),
        ("per_capita", 10.0, 0.0, 10_001),
        ("per_capita", 100.0, 0.0, 10_001),
        ("per_capita", 1_000.0, 0.0, 10_001),
        ("per_capita", 1_000.0, 0.0, 101),
        ("per_capita", 1_000.0, 0.0, None),
        ("per_capita", 10_000.0, 0.0, 10_001),
        ("per_capita", 100_000.0, 0.0, 10_001),
        ("per_capita", 1_000.0, 150.0, 10_001),
        ("per_capita", 1_000.0, 221.0, 10_001),
        ("per_capita", 1_000.0, 300.0, 10_001),
        ("per_capita", 1_000.0, 1_000.0, None),
        ("proportional", 0.0, 0.0, None),
        ("proportional", 1_000.0, 0.0, 10_001),
        ("none", 0.0, 0.0, None),
        ("none", 1_000.0, 0.0, 10_001),
        ("sign_flip", 0.0, 0.0, None),
        ("sign_flip", 1_000.0, 0.0, 10_001),
    ]
    for world, sigma, bias_b, n in cases:
        curve = analytic_plurality_curve(
            worlds[world],
            noise_sds=[sigma],
            n_voters=n,
            bias=(0.0, bias_b),
            welfare=Isoelastic(),
        )
        row = curve.iloc[0]
        vectors.append(
            {
                "world": world,
                "sigma": sigma,
                "bias_toward_b": bias_b,
                "n_voters": n,
                "p_tracked": round(float(row["p_tracked"]), 6),
                "share_optimal": round(float(row["expected_share_optimal"]), 6),
                "share_abstain": round(float(row["expected_share_abstain"]), 6),
                "mean_ranking_accuracy": (
                    None
                    if np.isnan(row["mean_ranking_accuracy"])
                    else round(float(row["mean_ranking_accuracy"]), 6)
                ),
            }
        )
    return vectors


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    gross = load_measured_electorate()
    worlds = build_worlds(gross)

    financed = worlds["per_capita"]
    gaussian_mean = float(np.average(financed.margins, weights=financed.weights))
    gaussian_sd = float(
        np.sqrt(
            np.average(
                (financed.margins - gaussian_mean) ** 2, weights=financed.weights
            )
        )
    )
    totals = gross.household_dollars()
    artifact = _data_dir() / f"{ARTIFACT_STEM}.parquet"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    ).stdout.strip()

    payload = {
        "provenance": {
            "generator": "democrasim scripts/export_web_assets.py",
            "democrasim_version": metadata.version("democrasim"),
            "democrasim_commit": commit,
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "constants": {
            "labels": list(gross.policy_labels),
            "welfare_optimal_index": int(np.argmax(Isoelastic().per_policy(financed))),
            "gross_totals_bn": [round(t / 1e9, 3) for t in totals],
            "levy_per_adult": [round(float(x), 2) for x in totals / gross.population],
            "population": round(gross.population),
            "n_rows": gross.n_voters,
            "mean_abs_margin_per_capita": round(
                float(np.average(np.abs(financed.margins), weights=financed.weights)),
                2,
            ),
        },
        "worlds": {name: compress_margins(world) for name, world in worlds.items()},
        "toys": {
            "gaussian": {
                "mean": round(gaussian_mean, 4),
                "sd": round(gaussian_sd, 2),
            },
            "homogeneous_margin": round(
                float(np.average(np.abs(financed.margins), weights=financed.weights)),
                2,
            ),
        },
        "test_vectors": test_vectors(worlds),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    pairs = {name: len(w["margins"]) for name, w in payload["worlds"].items()}
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); pairs per world: {pairs}")


if __name__ == "__main__":
    main()
