"""Load the committed measured-impacts artifact.

The artifact under ``democrasim/data/`` is *model output with provenance*,
not observed data: per-household policy impacts computed by the PolicyEngine
US microsimulation model on its certified Populace-backed microdata, one row
per voting-age adult. The sidecar ``.meta.json`` records exactly how it was
built (package versions, dataset, reform parameter dictionaries, totals) and
:func:`artifact_metadata` exposes it. Rebuild with ``democrasim build-data``
(requires the ``engine`` extra).
"""

import json
from importlib import resources
from pathlib import Path

import pandas as pd

from democrasim.electorate import Electorate

ARTIFACT_STEM = "us_2026_measured"

#: Columns that feed Electorate arrays; everything else becomes demographics.
_ARRAY_COLUMNS = frozenset({"weight", "base_income", "hh_adults"})


def _data_dir() -> Path:
    return Path(str(resources.files("democrasim") / "data"))


def artifact_metadata(stem: str = ARTIFACT_STEM) -> dict:
    """Provenance metadata recorded when the artifact was built."""
    path = _data_dir() / f"{stem}.meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no artifact metadata at {path}; run `democrasim build-data`"
        )
    return json.loads(path.read_text())


def load_measured_electorate(stem: str = ARTIFACT_STEM) -> Electorate:
    """The measured US electorate: real households, engine-computed impacts."""
    meta = artifact_metadata(stem)
    frame = pd.read_parquet(_data_dir() / f"{stem}.parquet")

    policy_columns = [p["column"] for p in meta["policies"]]
    labels = tuple(p["label"] for p in meta["policies"])
    demographic_columns = [
        c for c in frame.columns if c not in _ARRAY_COLUMNS and c not in policy_columns
    ]
    return Electorate(
        deltas=frame[policy_columns].to_numpy(dtype=float),
        weights=frame["weight"].to_numpy(dtype=float),
        base_income=frame["base_income"].to_numpy(dtype=float),
        hh_adults=frame["hh_adults"].to_numpy(dtype=float),
        policy_labels=labels,
        source=meta["source"],
        demographics=(frame[demographic_columns] if demographic_columns else None),
    )
