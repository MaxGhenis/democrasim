"""Command-line interface: demo, sweep, build-data.

``democrasim demo``
    A two-minute tour on the committed measured artifact: what the policies
    do, what welfare says, and how elections go at a few noise levels.

``democrasim sweep``
    Reproduce the headline experiments (accuracy curve, threshold, bias
    sweep, toy comparison, voting-rule comparison) and write tidy CSVs,
    figures, and a headline JSON under ``docs/``.

``democrasim build-data``
    Regenerate the measured artifact from the engine (needs the ``engine``
    extra); forwards arguments to ``democrasim.engine.build``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from democrasim.election import ElectionSpec
from democrasim.electorate import Electorate
from democrasim.experiments import accuracy_sweep, bias_sweep, find_threshold
from democrasim.perception import LinearGaussianPerception
from democrasim.toy import homogeneous_toy, moment_matched_toy
from democrasim.voting import Approval, InstantRunoff, Plurality
from democrasim.welfare import FinancingMode, Isoelastic, apply_financing

DEFAULT_NOISE_GRID = [0.0, *np.geomspace(10.0, 100_000.0, 13).tolist()]
DEFAULT_BIAS_GRID = np.linspace(0.0, 4_000.0, 11).tolist()
TRACKING_TARGET = 0.9


def _load_measured(financing: FinancingMode) -> Electorate:
    from democrasim.data import load_measured_electorate

    return apply_financing(load_measured_electorate(), financing)


def _mean_absolute_margin(electorate: Electorate) -> float:
    return float(np.average(np.abs(electorate.margins), weights=electorate.weights))


def demo(args: argparse.Namespace) -> None:
    from democrasim.data import artifact_metadata
    from democrasim.election import run_election

    meta = artifact_metadata()
    measured = _load_measured(args.financing)
    print(f"Source: {measured.source}")
    print(
        f"Financing: {args.financing} | rows: {measured.n_voters:,} adults "
        f"({measured.population / 1e6:.0f}M represented)\n"
    )
    for policy in meta["policies"]:
        print(f"{policy['label']}: {policy['description']}")
    print("\nImpact summary (net of financing):")
    print(measured.impact_summary().round(2).to_string(index=False))

    welfare = Isoelastic()
    by_policy = welfare.per_policy(measured)
    optimal = measured.policy_labels[int(np.argmax(by_policy))]
    print(f"\nIsoelastic (η=1) welfare ranks {optimal} first.")

    rng = np.random.default_rng(args.seed)
    print("\nElections (plurality, 10,001 sampled voters):")
    for noise_sd in (0.0, 1_000.0, 10_000.0):
        spec = ElectionSpec(
            perception=LinearGaussianPerception(noise_sd=noise_sd),
            welfare=welfare,
        )
        result = run_election(measured, spec, rng)
        winner = (
            "no winner" if result.winner < 0 else measured.policy_labels[result.winner]
        )
        print(
            f"  noise σ=${noise_sd:>8,.0f}: {winner:<12} "
            f"turnout {result.tally.turnout:.0%}, "
            f"tracked welfare: {result.tracked}"
        )


def sweep(args: argparse.Namespace) -> None:
    from democrasim import figures

    results_dir = Path(args.out) / "results"
    figures_dir = Path(args.out) / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    n_elections = 60 if args.quick else args.n_elections
    noise_grid = DEFAULT_NOISE_GRID[::2] if args.quick else DEFAULT_NOISE_GRID
    rng = np.random.default_rng(args.seed)

    measured = _load_measured(args.financing)
    worlds: dict[str, Electorate] = {
        "measured": measured,
        "toy_matched": moment_matched_toy(measured, rng),
        "toy_homogeneous": homogeneous_toy(
            margin=_mean_absolute_margin(measured), n=measured.n_voters
        ),
    }
    world_names = {
        "measured": "Measured households",
        "toy_matched": "Moment-matched Gaussian toy",
        "toy_homogeneous": "Homogeneous-stakes toy",
    }
    spec = ElectionSpec(
        perception=LinearGaussianPerception(),
        welfare=Isoelastic(),
        n_voters=args.n_voters,
    )

    # 1–2: accuracy sweeps and thresholds, per world.
    sweeps, thresholds = {}, {}
    for key, electorate in worlds.items():
        frame = accuracy_sweep(
            electorate,
            noise_sds=noise_grid,
            spec=spec,
            n_elections=n_elections,
            seed=args.seed,
        )
        frame.to_csv(results_dir / f"accuracy_sweep_{key}.csv", index=False)
        sweeps[key] = frame
        thresholds[key] = find_threshold(
            frame["mean_ranking_accuracy"],
            frame["p_tracked"],
            target=TRACKING_TARGET,
        )
        print(f"[sweep] {key}: threshold@{TRACKING_TARGET:.0%} = {thresholds[key]:.3f}")

    # 3: systematic bias toward the policy welfare ranks lower.
    welfare_by_policy = spec.welfare.per_policy(measured)
    toward = int(np.argmin(welfare_by_policy))
    bias_frame = bias_sweep(
        measured,
        bias_dollars=DEFAULT_BIAS_GRID,
        toward=toward,
        noise_sd=args.bias_noise_sd,
        spec=spec,
        n_elections=n_elections,
        seed=args.seed,
    )
    bias_frame.to_csv(results_dir / "bias_sweep_measured.csv", index=False)
    bias_tolerance = find_threshold(
        bias_frame["bias_dollars"], bias_frame[f"p_win_{toward}"], target=0.5
    )
    print(f"[sweep] bias tolerance (P(win)=50%): ${bias_tolerance:,.0f}")

    # 4: voting rules on the measured electorate.
    rule_sweeps, rule_names = [], []
    for rule_name, rule in (
        ("Plurality", Plurality()),
        ("Approval", Approval()),
        ("Instant runoff", InstantRunoff()),
    ):
        frame = accuracy_sweep(
            measured,
            noise_sds=noise_grid,
            spec=ElectionSpec(
                perception=LinearGaussianPerception(),
                rule=rule,
                welfare=Isoelastic(),
                n_voters=args.n_voters,
            ),
            n_elections=n_elections,
            seed=args.seed,
        )
        frame.to_csv(
            results_dir / f"rules_{rule_name.lower().replace(' ', '_')}.csv",
            index=False,
        )
        rule_sweeps.append(frame)
        rule_names.append(rule_name)

    headline = {
        "financing": args.financing,
        "welfare_metric": "isoelastic eta=1",
        "tracking_target": TRACKING_TARGET,
        "welfare_optimal": measured.policy_labels[int(np.argmax(welfare_by_policy))],
        # None = never reaches the target on the swept grid (strict-JSON null).
        "thresholds_ranking_accuracy": {
            key: (value if np.isfinite(value) else None)
            for key, value in thresholds.items()
        },
        "bias": {
            "toward": measured.policy_labels[toward],
            "noise_sd": args.bias_noise_sd,
            "tolerance_dollars": (
                bias_tolerance if np.isfinite(bias_tolerance) else None
            ),
        },
        "mean_absolute_margin_measured": _mean_absolute_margin(measured),
        "n_elections": n_elections,
        "n_voters": args.n_voters,
        "seed": args.seed,
    }
    (results_dir / "headline.json").write_text(json.dumps(headline, indent=2) + "\n")

    from democrasim.data import load_measured_electorate

    ordered = ["measured", "toy_matched", "toy_homogeneous"]
    saved = figures.save_figures(
        {
            # Gross engine deltas: "what the reforms do" (the share seeing
            # ≈$0 is the reform's reach). Financing enters the other figures.
            "impact_distribution": figures.impact_distribution(
                load_measured_electorate()
            ),
            "margin_distribution": figures.margin_distribution(
                [worlds[k] for k in ordered], [world_names[k] for k in ordered]
            ),
            "accuracy_curve": figures.accuracy_curve(
                [sweeps[k] for k in ordered],
                [world_names[k] for k in ordered],
                target=TRACKING_TARGET,
                thresholds=[thresholds[k] for k in ordered],
            ),
            "bias_curve": figures.bias_curve(
                [bias_frame],
                ["Measured households"],
                toward_label=measured.policy_labels[toward],
            ),
            "rule_comparison": figures.rule_comparison(rule_sweeps, rule_names),
        },
        figures_dir,
    )
    for path in saved:
        print(f"[sweep] wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="democrasim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo_parser = sub.add_parser("demo", help="quick tour on the artifact")
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument(
        "--financing",
        choices=("none", "per_capita", "proportional"),
        default="per_capita",
    )
    demo_parser.set_defaults(func=demo)

    sweep_parser = sub.add_parser("sweep", help="run headline experiments")
    sweep_parser.add_argument("--out", default="docs")
    sweep_parser.add_argument("--seed", type=int, default=42)
    sweep_parser.add_argument("--n-elections", type=int, default=240)
    sweep_parser.add_argument("--n-voters", type=int, default=10_001)
    sweep_parser.add_argument("--bias-noise-sd", type=float, default=1_000.0)
    sweep_parser.add_argument(
        "--financing",
        choices=("none", "per_capita", "proportional"),
        default="per_capita",
    )
    sweep_parser.add_argument("--quick", action="store_true")
    sweep_parser.set_defaults(func=sweep)

    build_parser = sub.add_parser(
        "build-data", help="regenerate the measured artifact (engine extra)"
    )
    build_parser.add_argument("rest", nargs=argparse.REMAINDER)
    build_parser.set_defaults(
        func=lambda args: __import__("democrasim.engine.build", fromlist=["main"]).main(
            args.rest
        )
    )

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
