import numpy as np
import pytest

from democrasim import (
    Electorate,
    Isoelastic,
    Utilitarian,
    apply_financing,
    welfare_optimal,
)


def two_voter_electorate(
    deltas: list[list[float]],
    incomes: list[float],
    weights: list[float] | None = None,
) -> Electorate:
    n = len(deltas)
    return Electorate(
        deltas=np.array(deltas, dtype=float),
        weights=np.array(weights if weights else [1.0] * n),
        base_income=np.array(incomes, dtype=float),
        hh_adults=np.ones(n),
        policy_labels=("Policy A", "Policy B"),
        source="TEST",
    )


class TestUtilitarian:
    def test_equals_household_dollars(self, small_electorate):
        np.testing.assert_allclose(
            Utilitarian().per_policy(small_electorate),
            small_electorate.household_dollars(),
        )

    def test_isoelastic_eta_zero_matches_utilitarian(self, small_electorate):
        linear = Isoelastic(eta=0.0).per_policy(small_electorate)
        np.testing.assert_allclose(
            linear, Utilitarian().per_policy(small_electorate), rtol=1e-12
        )


class TestIsoelastic:
    def test_concavity_prefers_gains_to_the_poor(self):
        # Policy A gives $1,000 to a $20k household; policy B gives $1,000
        # to a $200k household. Same dollars — utilitarian is indifferent,
        # any eta > 0 prefers A.
        electorate = two_voter_electorate(
            deltas=[[1_000.0, 0.0], [0.0, 1_000.0]],
            incomes=[20_000.0, 200_000.0],
        )
        utilitarian = Utilitarian().per_policy(electorate)
        assert utilitarian[0] == pytest.approx(utilitarian[1])
        for eta in (0.5, 1.0, 2.0):
            welfare = Isoelastic(eta=eta).per_policy(electorate)
            assert welfare[0] > welfare[1]
        assert welfare_optimal(Isoelastic(), electorate) == 0

    def test_income_floor_handles_nonpositive_income(self):
        electorate = two_voter_electorate(
            deltas=[[100.0, 0.0], [0.0, 100.0]],
            incomes=[-5_000.0, 50_000.0],
        )
        welfare = Isoelastic(eta=1.0).per_policy(electorate)
        assert np.all(np.isfinite(welfare))
        # The floored-income household still counts as the poorest.
        assert welfare[0] > welfare[1]

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError):
            Isoelastic(eta=-1.0)
        with pytest.raises(ValueError):
            Isoelastic(income_floor=0.0)


class TestFinancing:
    def test_none_is_identity(self, small_electorate):
        assert apply_financing(small_electorate, "none") is small_electorate

    def test_per_capita_is_budget_neutral(self, small_electorate):
        financed = apply_financing(small_electorate, "per_capita")
        np.testing.assert_allclose(financed.household_dollars(), [0.0, 0.0], atol=1e-9)

    def test_per_capita_burden_scales_with_adults(self, small_electorate):
        financed = apply_financing(small_electorate, "per_capita")
        burden = small_electorate.deltas - financed.deltas
        # Two-adult household bears twice the single-adult burden.
        np.testing.assert_allclose(burden[0], 2 * burden[2])

    def test_proportional_is_budget_neutral(self, small_electorate):
        financed = apply_financing(small_electorate, "proportional")
        np.testing.assert_allclose(financed.household_dollars(), [0.0, 0.0], atol=1e-9)

    def test_proportional_burden_scales_with_income(self, small_electorate):
        financed = apply_financing(small_electorate, "proportional")
        burden = small_electorate.deltas - financed.deltas
        # $90k household bears 1.5× the $60k household's burden.
        np.testing.assert_allclose(burden[2], 1.5 * burden[3])

    def test_unknown_mode_rejected(self, small_electorate):
        with pytest.raises(ValueError, match="unknown financing"):
            apply_financing(small_electorate, "magic")  # type: ignore[arg-type]

    def test_financing_flips_who_gains(self):
        # Policy B hands $10,000 to the rich household only. Once financed
        # per capita, the poor household must be a net loser under B.
        electorate = two_voter_electorate(
            deltas=[[0.0, 0.0], [0.0, 10_000.0]],
            incomes=[20_000.0, 500_000.0],
        )
        financed = apply_financing(electorate, "per_capita")
        assert financed.deltas[0, 1] < 0
        assert financed.deltas[1, 1] > 0
