"""Author and execute docs/demo.ipynb.

Run with: uv run python scripts/make_notebook.py
The notebook is written from source cells below, executed top to bottom
against the committed artifact, and saved with outputs so it renders on
GitHub without anyone needing to run it.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

OUT = Path(__file__).resolve().parents[1] / "docs" / "demo.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

CELLS = [
    md(
        "# Democrasim: elections on measured stakes\n"
        "\n"
        "This notebook walks the core loop: **real households**, two "
        "**actual encoded reforms**, engine-computed **true impacts**, one "
        "labeled **perception** assumption, and pluggable **voting rules** "
        "— then asks how accurately voters must perceive their own stakes "
        "before elections reliably pick the welfare-optimal policy.\n"
        "\n"
        "> **What this is and isn't.** A thought experiment about one "
        "mechanism (self-interested voting under misperception of "
        "engine-computed household impacts). It is not political science: "
        "real voters weigh values, identity, and information far beyond "
        "their own net income. It is never election prediction. Platforms "
        "are generic (Policy A / Policy B), not candidates or parties. "
        "Every number below is a model result, not observed data."
    ),
    code(
        "%matplotlib inline\n"
        "import numpy as np\n"
        "import democrasim as d\n"
        "\n"
        "rng = np.random.default_rng(42)\n"
        "meta = d.artifact_metadata()\n"
        "electorate = d.load_measured_electorate()\n"
        "print(electorate.source)\n"
        'print(f"{electorate.n_voters:,} adult rows representing "\n'
        '      f"{electorate.population / 1e6:.0f}M adults")\n'
        "for p in meta['policies']:\n"
        "    print(f\"\\n{p['label']}: {p['description']}\\n\"\n"
        "          f\"  engine total: ${p['total_household_dollars_bn']:.1f}B; \"\n"
        "          f\"{p['share_households_gaining']:.1%} of households gain\")"
    ),
    md(
        "## The measured inputs\n"
        "\n"
        "Each row is a voting-age adult from PolicyEngine's certified "
        "Populace-backed US microdata. The impact columns are the engine's "
        "answer to *\"how would this household's annual net income change "
        'under each policy?"* — including knock-on effects through other '
        "taxes and benefits, which is why they are worth measuring instead "
        "of inventing."
    ),
    code("electorate.impact_summary().round(2)"),
    md(
        "## Financing: closing the budget\n"
        "\n"
        "Engine deltas are *gross* — a tax cut shows only winners because "
        "the deficit lands on nobody. We make each policy budget-neutral "
        "with an explicit, labeled rule (default: every adult bears an "
        "equal share of its cost). Net of financing, total dollars are ~0 "
        "for both policies by construction, so the welfare ranking is "
        "purely distributional — and an inequality-averse metric (isoelastic, "
        "η=1) does the ranking."
    ),
    code(
        "financed = d.apply_financing(electorate, 'per_capita')\n"
        "welfare = d.Isoelastic()\n"
        "by_policy = welfare.per_policy(financed)\n"
        "optimal = financed.policy_labels[int(np.argmax(by_policy))]\n"
        "for label, value in zip(financed.policy_labels, by_policy):\n"
        "    print(f'{label}: isoelastic welfare change {value:+.4g}')\n"
        "print(f'\\nWelfare-optimal policy: {optimal}')\n"
        "financed.impact_summary().round(2)"
    ),
    md(
        "## What voters actually have at stake\n"
        "\n"
        "The margin — how much better one policy is than the other *for "
        "your household* — is the signal perception noise competes with. "
        "On measured data most adults have almost nothing at stake and a "
        "minority has thousands of dollars at stake; the invented worlds "
        "the old model used had nothing like this shape."
    ),
    code(
        "from democrasim import figures, toy\n"
        "\n"
        "matched = toy.moment_matched_toy(financed, rng)\n"
        "mean_stake = float(np.average(np.abs(financed.margins),\n"
        "                              weights=financed.weights))\n"
        "homogeneous = toy.homogeneous_toy(margin=mean_stake,\n"
        "                                  n=financed.n_voters)\n"
        "figures.margin_distribution(\n"
        "    [financed, matched, homogeneous],\n"
        "    ['Measured households', 'Moment-matched Gaussian toy',\n"
        "     'Homogeneous-stakes toy'],\n"
        ");"
    ),
    md(
        "## One election\n"
        "\n"
        "Sample 10,001 adults (probability ∝ survey weight), let each "
        "perceive their own household's stakes through the perception "
        "model, vote by plurality (perceived indifference abstains), and "
        "grade the winner against the welfare ranking.\n"
        "\n"
        "Watch the zero-noise row: with financing on, the election is "
        "decided by the ~three-quarters of adults whose entire stake is "
        "their share of the two policies' residual cost gap (about $4 a "
        "year) — a quantity unrelated to the welfare ranking, so perfect "
        "information makes the outcome welfare-*independent*. Here the "
        "gap's sign favors the policy the metric ranks lower. Moderate "
        "noise turns those near-indifferent votes into fair coins that "
        "cancel, letting the high-stakes minority decide — misperception "
        "can *help*. The findings note shows the flip side: reverse the "
        "cost gap's sign, zero it exactly, or let trivial stakes abstain, "
        "and perfect accuracy tracks perfectly."
    ),
    code(
        "for noise_sd in (0.0, 1_000.0, 10_000.0):\n"
        "    spec = d.ElectionSpec(\n"
        "        perception=d.LinearGaussianPerception(noise_sd=noise_sd),\n"
        "        welfare=welfare,\n"
        "    )\n"
        "    result = d.run_election(financed, spec, rng)\n"
        "    winner = ('no winner' if result.winner < 0\n"
        "              else financed.policy_labels[result.winner])\n"
        "    accuracy = d.ranking_accuracy(spec.perception, financed)\n"
        "    print(f'σ=${noise_sd:>8,.0f}  ranking accuracy {accuracy:.2f}  '\n"
        "          f'→ {winner:<12} turnout {result.tally.turnout:.0%}  '\n"
        "          f'tracked: {result.tracked}')"
    ),
    md(
        "## The headline: accuracy → welfare tracking\n"
        "\n"
        "Sweep perception noise and plot the share of elections electing "
        "the ballot-best policy against the mean probability that a voter "
        "correctly ranks the two policies for their own household (0.5 = "
        "coin flip). Two caveats the findings note unpacks: every curve "
        "here is specific to the sampled electorate size (thresholds "
        "tighten toward 0.5 as n grows — Condorcet jury logic; "
        "`democrasim.analytic_plurality_curve` gives any n in closed "
        "form), and the accuracy axis is a derived coordinate, not a "
        "sufficient statistic. On measured stakes the curve is **not "
        "monotone**: tracking is essentially perfect across a wide band "
        "of moderate accuracy, and at perfect accuracy the outcome is "
        "decided by the welfare-irrelevant *sign* of the small residual "
        "cost gap between the two policies — the homogeneous toy is the "
        "jury theorem, and the Gaussian toy hovers near a coin flip at "
        "this n. This cell runs a reduced grid to stay quick; "
        "`democrasim sweep` reproduces the full version behind "
        "`docs/findings.md`."
    ),
    code(
        "noise_grid = [0.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0,\n"
        "              30_000.0, 100_000.0]\n"
        "spec = d.ElectionSpec(perception=d.PERFECT_PERCEPTION,\n"
        "                      welfare=welfare, n_voters=5_001)\n"
        "sweeps = {}\n"
        "for name, world in [('measured', financed), ('matched', matched),\n"
        "                    ('homogeneous', homogeneous)]:\n"
        "    sweeps[name] = d.accuracy_sweep(world, noise_grid, spec=spec,\n"
        "                                    n_elections=80, seed=7)\n"
        "thresholds = {name: d.find_threshold(s['mean_ranking_accuracy'],\n"
        "                                     s['p_tracked'], target=0.9)\n"
        "              for name, s in sweeps.items()}\n"
        "print('accuracy needed for 90% welfare tracking:')\n"
        "for name, value in thresholds.items():\n"
        "    print(f'  {name:>12}: {value:.3f}' if np.isfinite(value)\n"
        "          else f'  {name:>12}: never reaches 90%')\n"
        "figures.accuracy_curve(\n"
        "    [sweeps['measured'], sweeps['matched'], sweeps['homogeneous']],\n"
        "    ['Measured households', 'Moment-matched Gaussian toy',\n"
        "     'Homogeneous-stakes toy'],\n"
        "    target=0.9,\n"
        "    thresholds=[thresholds['measured'], thresholds['matched'],\n"
        "                thresholds['homogeneous']],\n"
        ");"
    ),
    md(
        "## Systematic bias\n"
        "\n"
        "Noise averages out across voters; bias does not. Give every voter "
        "a rosy view of the welfare-inferior policy — worth $b more per "
        "year than it truly is — and ask how many perceived dollars it "
        "takes to flip the election."
    ),
    code(
        "welfare_by_policy = welfare.per_policy(financed)\n"
        "toward = int(np.argmin(welfare_by_policy))\n"
        "bias_frame = d.bias_sweep(financed,\n"
        "                          bias_dollars=np.linspace(0, 4_000, 9),\n"
        "                          toward=toward, noise_sd=1_000.0,\n"
        "                          spec=spec, n_elections=80, seed=7)\n"
        "tolerance = d.find_threshold(bias_frame['bias_dollars'],\n"
        "                             bias_frame[f'p_win_{toward}'],\n"
        "                             target=0.5)\n"
        "print(f'bias flipping the median election: '\n"
        "      f'${tolerance:,.0f}/voter/year of perceived advantage')\n"
        "figures.bias_curve([bias_frame], ['Measured households'],\n"
        "                   toward_label=financed.policy_labels[toward]);"
    ),
    md(
        "## Where to next\n"
        "\n"
        "- `docs/findings.md` — the full toy-vs-measured comparison.\n"
        "- `democrasim.PerceptionModel` — the interface a survey-estimated "
        "misperception distribution will drop into.\n"
        "- `democrasim.Approval` / `democrasim.InstantRunoff` — the "
        "mechanism-comparison surface.\n"
        "\n"
        "*All results above are simulation outputs from the model described "
        "in the README, under the stated perception assumption — not "
        "measurements of any real electorate.*"
    ),
]


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = CELLS
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    # Pin the "python3" kernelspec to THIS interpreter via a scratch
    # JUPYTER_DATA_DIR — user-level kernelspecs with the same name may point
    # at unrelated virtualenvs, and nbclient resolves by name.
    with tempfile.TemporaryDirectory() as tmp:
        spec_dir = Path(tmp) / "kernels" / "python3"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-Xfrozen_modules=off",
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Python 3",
                    "language": "python",
                }
            )
        )
        os.environ["JUPYTER_DATA_DIR"] = tmp
        client = NotebookClient(
            notebook,
            timeout=1_200,
            resources={"metadata": {"path": str(OUT.parent)}},
        )
        client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
