# Endogenous platforms: the position game on measured stakes

The fixed-platform findings ([findings.md](findings.md)) grade elections
against two given policies. This layer makes the platforms choices: two
candidates pick positions to maximize their own expected objectives,
knowing what the opponent picks, and the same perception-mediated
electorate decides who wins. Every equilibrium below is exact — pure
Nash by exhaustive enumeration — and every number regenerates with
`uv run python scripts/strategic_experiments.py`.

## The game

**Policy space.** A position is a pair of intensities ``(α, β)`` on the
two measured incidence vectors: α scales the CTC increment (α=1 is the
full $2,200→$3,200 change), β scales the top-rate cap depth (β=1 caps
both top brackets at 34%). Positions produce gross household deltas
``α·Δ_A + β·Δ_B``, financed per capita. Under per-capita financing net
stakes are exactly linear in ``(α, β)``, so the whole space runs on the
two committed delta columns — the status quo ``(0, 0)`` included. The
default grid is 21×21 over ``[0, 1]²``.

**Interpolation is measured, not assumed.** Within-family linearity is
an approximation, so two engine runs at the midpoints price it
([results/interpolation_validation.json](results/interpolation_validation.json)):
the rate axis is linear to 0.13% of totals (99.95% of households within
$1); the CTC axis carries real phase-in convexity — 5.0% of totals, 96%
of households within $1, mean absolute error $9. The grid stays inside
the validated range.

**Candidates.** Each candidate is a household from the data with an
objective over enacted positions, in dollars per year:
``selfish_weight · own household net delta + (1 − selfish_weight) ·
societal dollar value``, where the societal value is the position's
change in equally-distributed-equivalent (EDE) household income under
the candidate's own inequality aversion. Both terms are dollars, so a
candidate at weight 0.5 trades a dollar of their own for a dollar of
society-wide EDE income per household — no normalization, and the same
utility family the electorate can carry ([heterogeneity.md](heterogeneity.md)).
An optional office rent adds a fixed dollar win bonus. Expected utility
is ``P(win)·U(own position) + (1−P(win))·U(opponent's)``.

The headline candidates are selected deterministically *by measured
stake*, and the selection is asserted against the committed deltas:
Candidate 1 is a two-adult household with three children at the median
income of such households ($109,693; gross Policy-A stake +$1,133).
Candidate 2 is the household at the weighted median income of Policy B's
gross winners ($578,540, childless; gross Policy-B stake +$4,655) — an
income-percentile proxy fails here, because the childless 95th
percentile sits below the capped brackets and would *lose* from
Policy B.

**Election.** P(win) uses the same probit closed form as the
fixed-platform model, at 10,001 sampled voters. For the baseline
self-interested electorate it depends only on the difference of
positions, so one (2G−1)² table prices all G⁴ profile comparisons and
exact pure-Nash enumeration costs about a second per configuration; any
mixture of voter types drops into the same tables
([heterogeneity.md](heterogeneity.md)).

## Results

Mean enacted position across equilibria, written α/β (Policy-A and
Policy-B intensity), with candidate proposals from best-response
dynamics in parentheses
([results/strategic_equilibria.csv](results/strategic_equilibria.csv);
the welfare-optimal position under η=1 is the status quo, consistent
with the fixed-platform finding that both financed endpoint policies
score below it):

| Candidates | σ = $0 | σ = $200 | σ = $1,000 | σ = $10,000 | σ = $30,000 |
|---|---|---|---|---|---|
| Office-seekers (rent only) | status quo | status quo | status quo | status quo | status quo |
| Both societal (η=1) | status quo | status quo | status quo | status quo | status quo |
| Both selfish | status quo | .01/0 (.05/0) | .19/.05 (.35/.10) | .65/.25 (1.0/.70) | .55/.45 (1.0/1.0) |

![Equilibrium platform intensity vs noise](figures/strategic_positions.png)

Three regularities:

1. **Perfect information disciplines platforms to the status quo.** At
   σ ≤ $10 every equilibrium enacts ``(0, 0)``, whatever the candidates
   want. The fixed-platform knife-edge — the no-stakes mass voting its
   levy sliver — becomes an undercutting force: any program hands the
   opponent a cheaper-platform win, so competition converges on
   proposing nothing. Under this welfare metric that *is* the optimum:
   with endogenous platforms, perfect information produces the
   welfare-best outcome through competition, inverting the
   fixed-platform result where it produced welfare-independence.
2. **Noise relaxes the discipline, and self-interest fills the gap.**
   As σ grows, both candidates escalate their own programs — the
   child-household candidate from a doomed α=0.05 proposal at σ=$50
   (enacted: still zero) to the full CTC program by σ=$10,000, the
   rate-cap candidate from β=0.05 at σ=$1,000 to the full cap by
   σ=$30,000. Under fixed platforms, moderate noise rescued welfare
   tracking; with endogenous platforms the same noise is what lets
   welfare-negative programs through. Whether misperception helps
   depends on who sets the agenda. The selfish weight barely matters:
   0.5 behaves almost like 1.0, because own stakes ($1,133 and $4,655
   gross) dwarf the societal values (−$169 and −$379 of EDE per
   household at full intensity) — only near-total societal weight
   changes candidate behavior.
3. **Electoral competition prices constituency breadth — and noise
   erodes the price.** The CTC program's gross winners are a fifth of
   adults; the rate cap's are 3%. At σ=$1,000 that breadth gap keeps
   enacted intensity 4-to-1 in Policy A's favor (0.19 vs 0.05). By
   σ=$30,000 both candidates run their full programs and the election
   is a near-coin flip (win probability 0.545 vs 0.455): enough noise
   erases the electorate's ability to distinguish a broad program from
   a narrow one. The breadth filter is a head-count, indifferent to the
   welfare metric — and it only works when voters can see their stakes.

## Scope

Two candidates, one shot, pure strategies on a grid, a shared 2D policy
space, and objectives in one dollar family — all labeled choices, all
cheap to vary. Mixed-strategy equilibria are not computed (in the
baseline sweep every configuration has a pure equilibrium; heterogeneous
electorates can empty the pure set — [heterogeneity.md](heterogeneity.md)
reports where, and best-response dynamics there). The equilibrium
multiplicity at low noise is tie-driven: losing positions with identical
payoffs proliferate profiles; the *enacted* position is the invariant
summary. Candidate behavior beyond this objective family — dynamics,
entry, primaries, commitment problems — is out of scope, as is any claim
about real candidates.

## Reproduction

```bash
uv run python scripts/strategic_experiments.py     # equilibria, office-seeker variant, figure
uv run python scripts/validate_interpolation.py    # two engine runs (~16 min) + error report
uv run pytest tests/test_strategic.py              # behavioral tests, incl. voter-type tables
```
