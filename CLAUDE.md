# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

Democrasim asks one question: **do elections select the welfare-maximizing
policy when voters misperceive how policies would affect them?** Everything
except perception is measured:

- Electorate = real US households (PolicyEngine's certified Populace-backed
  microdata), one voting-age adult per row.
- Platforms = actual encoded tax reforms with opposite incidence, labeled
  generically (Policy A / Policy B — never real candidates or parties).
- True impacts = engine-computed per-household net income deltas.
- Perception = the one labeled assumption, behind the `PerceptionModel`
  interface, designed for a survey-measured distribution to drop in later.

This is a thought experiment about one mechanism, not political science and
never election prediction. Keep platform labels generic and neutral.

## Commands

```bash
uv sync --group dev              # core dev environment (no engine)
uv run pytest                    # test suite (engine tests skip if absent)
uv run ruff check . && uv run ruff format .
uv run democrasim sweep          # experiments on the committed artifact
uv sync --extra engine --group dev   # only to rebuild the measured dataset
uv run democrasim build-data     # regenerate democrasim/data artifact
```

## Architecture

```
democrasim/
  electorate.py   # Electorate: numpy arrays (deltas, weights, hh_adults, ...)
  perception.py   # PerceptionModel protocol + LinearGaussianPerception
  voting.py       # Plurality / Approval / InstantRunoff over perceived deltas
  welfare.py      # Utilitarian / Isoelastic functionals + financing modes
  election.py     # one election: sample -> perceive -> vote -> compare to welfare
  experiments.py  # accuracy sweeps, threshold finder, bias sweeps
  toy.py          # moment-matched Gaussian comparator (the old model's world)
  data.py         # load the committed measured artifact
  engine/build.py # regenerates the artifact (subprocess per simulation)
  cli.py          # argparse CLI: build-data / sweep / demo
```

## House rules

- Python 3.14, `uv` (never pip), pytest with behavioral tests, full typing.
- No Streamlit anywhere. Demo surface = notebook + CLI.
- Engine work: never compute aggregates from raw weight arrays — use
  MicroSeries (`.calc(...).sum()`); the artifact builder validates its
  extracted arrays against MicroSeries aggregates before writing.
- Reform dicts must carry explicit start AND end dates
  (`{"param": {"2026-01-01.2100-12-31": value}}`).
- Run at most one engine simulation per process (subprocess-per-scenario);
  two sims in one kernel fragments the allocator and OOMs.
- Never present simulation output as observed data; the artifact carries
  provenance metadata and README/notebook state the data source.
- Sentence case for headings.
