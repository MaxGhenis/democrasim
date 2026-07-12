# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this is

Democrasim asks one question: **do elections select the welfare-maximizing
policy when voters misperceive how policies would affect them?** The inputs
are measured; the behavioral layers are labeled assumptions:

- Electorate = US households from PolicyEngine's certified,
  population-calibrated Populace microdata, one voting-age adult per row.
- Platforms = actual encoded tax reforms with opposite incidence, labeled
  generically (Policy A / Policy B — never real candidates or parties).
- True impacts = engine-computed per-household net income deltas.
- Perception = the headline assumption, behind the `PerceptionModel`
  interface, designed for a survey-measured distribution to drop in later;
  financing, welfare metric, and electorate size are labeled model layers,
  stress-tested in docs/findings.md.

This is a thought experiment about one mechanism, not political science and
never election prediction. Keep platform labels generic and neutral.

## Commands

```bash
uv sync --group dev              # core dev environment (no engine)
uv run pytest                    # test suite (engine tests skip if absent)
uv run ruff check . && uv run ruff format .
uv run democrasim sweep          # headline experiments -> docs/results + figures
uv run python scripts/robustness.py    # stress rows behind docs/findings.md
uv run python scripts/descriptives.py  # descriptive numbers behind findings §1-3
uv run python scripts/make_notebook.py # re-execute docs/demo.ipynb
uv run python scripts/strategic_experiments.py  # Nash equilibria behind docs/strategic.md
uv run python scripts/heterogeneous_experiments.py  # mixed-motive voters behind docs/heterogeneity.md
uv run python scripts/rules_experiments.py  # three-option rules comparison behind docs/rules.md
uv sync --extra engine --group dev   # only to rebuild the measured dataset
uv run democrasim build-data     # regenerate democrasim/data artifact
```

Every number quoted in README.md, docs/findings.md, docs/strategic.md,
docs/heterogeneity.md, or docs/rules.md must trace to a file in
docs/results/ produced by one of the commands above, and
tests/test_findings_regression.py pins the artifact facts the findings rest
on (most fragilely: the sign of the cost gap between the two policies). If a
rebuild trips those tests, the findings note must be re-derived, not patched.

## Architecture

```
democrasim/
  electorate.py   # Electorate: numpy arrays (deltas, weights, hh_adults, ...)
  perception.py   # PerceptionModel protocol + LinearGaussianPerception
  voting.py       # Plurality / Approval / Score / STAR / InstantRunoff over perceived dollars
  welfare.py      # Utilitarian / Isoelastic functionals + dollar EDE + financing
  preferences.py  # VoterType mixtures: selfish/societal motives on one dollar scale
  election.py     # one election: sample -> perceive -> vote -> compare to welfare
  experiments.py  # accuracy sweeps, threshold finder, bias sweeps
  strategic.py    # endogenous platforms: exact Nash on the 2D policy space
  toy.py          # moment-matched Gaussian comparator (labeled synthetic worlds)
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
