"""Generate docs/results/descriptives.json from the committed artifact.

Every descriptive quantity quoted in docs/findings.md §1–§3 (levies, margin
shares, preference splits, incidence by decile, welfare values) comes from
this script. Run: uv run python scripts/descriptives.py
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np

import democrasim as d
from democrasim.data import ARTIFACT_STEM, _data_dir
from democrasim.electorate import Electorate

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"


def provenance() -> dict:
    artifact = _data_dir() / f"{ARTIFACT_STEM}.parquet"
    return {
        "generator": "scripts/descriptives.py",
        "democrasim_version": metadata.version("democrasim"),
        "artifact": ARTIFACT_STEM,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }


def dollars_by_top_decile(electorate: Electorate) -> dict[str, float]:
    """Share of each policy's household-once dollars in the top decile.

    Deciles are defined over baseline household net income, weighted by
    household mass (weight / hh_adults) — i.e. household-weighted deciles,
    matching the builder's --report definition.
    """
    household_mass = electorate.weights / electorate.hh_adults
    order = np.argsort(electorate.base_income)
    cum = np.cumsum(household_mass[order])
    decile = np.empty(electorate.n_voters, dtype=int)
    decile[order] = np.minimum((10 * cum / cum[-1]).astype(int) + 1, 10)
    out = {}
    for j, label in enumerate(electorate.policy_labels):
        dollars = household_mass * electorate.deltas[:, j]
        total = dollars.sum()
        out[label] = float(dollars[decile == 10].sum() / total) if total else 0.0
    return out


def main() -> None:
    gross = d.load_measured_electorate()
    fin = d.apply_financing(gross, "per_capita")
    w = fin.weights

    def share(mask, wt=w) -> float:
        return float(wt[mask].sum() / wt.sum())

    m = fin.margins
    gm = gross.margins
    totals = gross.household_dollars()
    levy = totals / gross.population
    has_children = fin.demographics["n_children"].to_numpy() > 0

    eta_rankings = {}
    for eta in (0.5, 1.0, 2.0, 3.0, 5.0):
        values = d.Isoelastic(eta=eta).per_policy(fin)
        eta_rankings[str(eta)] = fin.policy_labels[int(np.argmax(values))]

    descriptives = {
        "provenance": provenance(),
        "gross_totals_bn": [round(t / 1e9, 3) for t in totals],
        "levy_per_adult": [round(x, 4) for x in levy],
        "levy_gap_per_adult": round(float(levy[0] - levy[1]), 4),
        "gross_dollars_top_decile_share": dollars_by_top_decile(gross),
        "financed_margin": {
            "mean_abs": round(float(np.average(np.abs(m), weights=w)), 2),
            "share_abs_below_10": round(share(np.abs(m) < 10), 4),
            "share_abs_below_25": round(share(np.abs(m) < 25), 4),
            "share_abs_below_100": round(share(np.abs(m) < 100), 4),
            "share_abs_below_1000": round(share(np.abs(m) < 1000), 4),
            "share_prefer_A": round(share(m > 0), 4),
            "share_prefer_B": round(share(m < 0), 4),
        },
        "gross_margin": {
            "share_zero_both": round(
                share(
                    (np.abs(gross.deltas[:, 0]) < 1) & (np.abs(gross.deltas[:, 1]) < 1)
                ),
                4,
            ),
            "share_prefer_A": round(share(gm > 1), 4),
            "share_prefer_B": round(share(gm < -1), 4),
            "share_indifferent": round(share(np.abs(gm) <= 1), 4),
        },
        "high_stakes": {
            "share_margin_above_500": round(share(m > 500), 4),
            "kids_share_of_pro_A_500": round(
                float(w[(m > 500) & has_children].sum() / max(w[m > 500].sum(), 1.0)),
                4,
            ),
            "share_margin_below_minus_500": round(share(m < -500), 4),
        },
        "welfare_isoelastic_eta1": [
            float(f"{x:.6g}") for x in d.Isoelastic().per_policy(fin)
        ],
        "welfare_utilitarian_financed": [
            float(f"{x:.4g}") for x in d.Utilitarian().per_policy(fin)
        ],
        "isoelastic_optimal_by_eta": eta_rankings,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "descriptives.json").write_text(json.dumps(descriptives, indent=2) + "\n")
    print(json.dumps(descriptives, indent=2))


if __name__ == "__main__":
    main()
