"""Democrasim: do elections track welfare when voters misperceive impacts?

An election simulation with measured inputs: households from certified,
population-calibrated US microdata (Populace), actual encoded reforms with
opposite incidence, and engine-computed per-household impacts.
Perception — how accurately voters see their own stakes — is the headline
assumption, held behind an interface a measured distribution can drop into;
financing, the welfare metric, and electorate size are further labeled,
stress-tested model layers.

A thought experiment about one mechanism. Not political science, and never
election prediction.
"""

from democrasim.data import artifact_metadata, load_measured_electorate
from democrasim.election import ElectionResult, ElectionSpec, run_election
from democrasim.electorate import Electorate, weighted_quantile
from democrasim.experiments import (
    accuracy_sweep,
    analytic_plurality_curve,
    bias_sweep,
    find_threshold,
    run_elections,
    summarize_elections,
)
from democrasim.perception import (
    PERFECT_PERCEPTION,
    GroupedPerception,
    LinearGaussianPerception,
    PerceptionModel,
    ranking_accuracy,
)
from democrasim.toy import homogeneous_toy, moment_matched_toy
from democrasim.voting import (
    NO_WINNER,
    Approval,
    InstantRunoff,
    Plurality,
    Tally,
    VotingRule,
)
from democrasim.welfare import (
    Isoelastic,
    Utilitarian,
    WelfareMetric,
    apply_financing,
    welfare_optimal,
)

__version__ = "0.2.0"

__all__ = [
    "NO_WINNER",
    "PERFECT_PERCEPTION",
    "Approval",
    "ElectionResult",
    "ElectionSpec",
    "Electorate",
    "GroupedPerception",
    "InstantRunoff",
    "Isoelastic",
    "LinearGaussianPerception",
    "PerceptionModel",
    "Plurality",
    "Tally",
    "Utilitarian",
    "VotingRule",
    "WelfareMetric",
    "accuracy_sweep",
    "analytic_plurality_curve",
    "apply_financing",
    "artifact_metadata",
    "bias_sweep",
    "find_threshold",
    "homogeneous_toy",
    "load_measured_electorate",
    "moment_matched_toy",
    "ranking_accuracy",
    "run_election",
    "run_elections",
    "summarize_elections",
    "weighted_quantile",
    "welfare_optimal",
]
