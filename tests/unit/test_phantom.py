import pytest
from datetime import date, datetime
import random

import numpy as np

from smart_mediator_assignment.algorithm.phantom import (
    generate_phantom_cases,
    estimate_case_arrivals,
)
from tests.fixtures import (
    SCENARIO1_AVG_CASE_RATE,
    SCENARIO1_AVG_P_VAL,
    SCENARIO1_MED_BY_CRT_CASE_TYPE,
)

_ARGS = dict(
    current_day=date(2023, 1, 1), time_horizon=5,
    avg_case_rate={"Family group": {"MILIMANI": 2.0}},
    avg_p_val_by_crt_case_type={("Family group", "MILIMANI"): 0.5},
    med_by_court_case_type={"MILIMANI": {"Family group": [1, 2]}},
    court_stations=["MILIMANI"], case_types=["Family group"],
)


class TestGeneratePhantomCases:
    """Tests for phantom case generation."""

    def test_generates_phantom_cases(self):
        """Test that phantom cases are generated."""
        phantom_cases, next_id = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        assert len(phantom_cases) > 0
        assert next_id < -1

    def test_phantom_ids_are_negative(self):
        """Test that all phantom case IDs are negative."""
        phantom_cases, _ = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        for case in phantom_cases:
            assert case.id < 0

    def test_phantom_dates_are_future(self):
        """Test that phantom cases have future dates."""
        current_day = date(2023, 1, 1)
        phantom_cases, _ = generate_phantom_cases(
            current_day=current_day,
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        for case in phantom_cases:
            assert case.referral_date >= current_day

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        args = dict(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        cases1, id1 = generate_phantom_cases(**args)
        cases2, id2 = generate_phantom_cases(**args)

        assert len(cases1) == len(cases2)
        assert id1 == id2
        for c1, c2 in zip(cases1, cases2):
            assert c1.id == c2.id
            assert c1.court_station == c2.court_station

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results over many trials."""
        args = dict(
            current_day=date(2023, 1, 1),
            time_horizon=30,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
        )

        cases1, _ = generate_phantom_cases(**args, seed=42)
        cases2, _ = generate_phantom_cases(**args, seed=123)

        different_count = len(cases1) != len(cases2)
        if not different_count and len(cases1) > 0:
            different_dates = any(
                c1.referral_date != c2.referral_date or c1.court_station != c2.court_station
                for c1, c2 in zip(cases1, cases2)
            )
            different_count = different_dates

        assert different_count or len(cases1) == 0

    def test_zero_time_horizon(self):
        """Test that zero time horizon produces no cases."""
        phantom_cases, next_id = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=0,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        assert len(phantom_cases) == 0
        assert next_id == -1

    def test_uses_avg_p_val(self):
        """Test that p-values come from avg_p_val mapping."""
        phantom_cases, _ = generate_phantom_cases(
            current_day=date(2023, 1, 1),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        for case in phantom_cases:
            assert case.p_value == pytest.approx(0.4, abs=0.01)

    def test_datetime_support(self):
        """Test that datetime objects work as well as date objects."""
        phantom_cases, _ = generate_phantom_cases(
            current_day=datetime(2023, 1, 1, 10, 30),
            time_horizon=10,
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            avg_p_val_by_crt_case_type=SCENARIO1_AVG_P_VAL,
            med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            starting_id=-1,
            seed=42,
        )

        assert len(phantom_cases) > 0

    def test_same_seed_is_reproducible(self):
        a, _ = generate_phantom_cases(**_ARGS, seed=7)
        b, _ = generate_phantom_cases(**_ARGS, seed=7)
        assert [(c.id, c.referral_date) for c in a] == [(c.id, c.referral_date) for c in b]

    def test_does_not_disturb_global_rng(self):
        np.random.seed(123); random.seed(123)
        exp_np, exp_py = np.random.rand(), random.random()
        np.random.seed(123); random.seed(123)
        generate_phantom_cases(**_ARGS, seed=999)
        assert np.random.rand() == exp_np and random.random() == exp_py


class TestEstimateCaseArrivals:
    """Tests for case arrival estimation."""

    def test_estimates_arrivals(self):
        """Test that arrivals are estimated correctly."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            days=1,
        )

        assert ("Family group", "MILIMANI") in estimates
        assert ("Family group", "KAKAMEGA") in estimates
        assert estimates[("Family group", "MILIMANI")] == pytest.approx(0.055)
        assert estimates[("Family group", "KAKAMEGA")] == pytest.approx(0.05)

    def test_multi_day_estimation(self):
        """Test estimation over multiple days."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["MILIMANI", "KAKAMEGA"],
            case_types=["Family group"],
            days=10,
        )

        assert estimates[("Family group", "MILIMANI")] == pytest.approx(0.55)
        assert estimates[("Family group", "KAKAMEGA")] == pytest.approx(0.5)

    def test_missing_case_type(self):
        """Test that missing case types are skipped."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["MILIMANI"],
            case_types=["Unknown Type"],
            days=1,
        )

        assert len(estimates) == 0

    def test_missing_court_station(self):
        """Test that missing court stations are skipped."""
        estimates = estimate_case_arrivals(
            avg_case_rate=SCENARIO1_AVG_CASE_RATE,
            court_stations=["Unknown Station"],
            case_types=["Family group"],
            days=1,
        )

        assert len(estimates) == 0


from collections import Counter

from smart_mediator_assignment.algorithm.phantom import (
    ArrivalPool,
    build_arrival_pool,
    generate_phantom_cases_from_pool,
)
from smart_mediator_assignment.algorithm.va_estimation import VAModel
from smart_mediator_assignment.core import SimpleCase

_MODEL = VAModel(
    intercept=0.5,
    coefficients={
        'appt_month': {1.0: 0.0, 3.0: 0.02},
        'court_station': {'AAAMilimani': 0.0, 'KAKAMEGA': 0.05},
        'referral_mode': {'Referred by Court': 0.0, 'Request by Parties': 0.04},
        'highcourt': {0.0: 0.0, 1.0: 0.01},
    },
)


def _arrival(case_id, referred, station="MILIMANI", referral_mode="Referred by Court",
             court_type="Magistrate Court", case_type="Divorce and Separation"):
    return SimpleCase(id=case_id, case_type=case_type, court_station=station,
                      referral_date=referred, referral_mode=referral_mode, court_type=court_type)


class TestBuildArrivalPool:
    def test_keeps_only_the_window_up_to_as_of(self):
        cases = [
            _arrival(1, date(2025, 6, 30)),    # exactly window_days before as_of: excluded
            _arrival(2, date(2025, 7, 1)),
            _arrival(3, date(2025, 12, 29)),   # on as_of: included
            _arrival(4, date(2025, 12, 30)),   # after as_of: excluded
            SimpleCase(id=5, case_type="Civil Cases", court_station="MILIMANI", referral_date=None),
        ]
        pool = build_arrival_pool(cases, as_of=date(2025, 12, 29), window_days=182)
        assert len(pool.records) == 2
        assert pool.daily_rate == pytest.approx(2 / 182)

    def test_record_keeps_raw_case_type_and_court_type(self):
        pool = build_arrival_pool([_arrival(1, date(2025, 12, 1), court_type="Kadhi Court")],
                                  as_of=date(2025, 12, 29))
        assert pool.records[0] == {
            'case_type': "Divorce and Separation", 'court_station': "MILIMANI",
            'referral_mode': "Referred by Court", 'court_type': "Kadhi Court",
        }


class TestGeneratePhantomCasesFromPool:
    _POOL = ArrivalPool(
        records=(
            {'case_type': "Divorce and Separation", 'court_station': "KAKAMEGA",
             'referral_mode': "Request by Parties", 'court_type': "High Court"},
        ),
        daily_rate=2.0,
    )

    def _generate(self, seed=7, **kwargs):
        args = dict(current_day=date(2026, 3, 9), time_horizon=10, pool=self._POOL,
                    va_model=_MODEL, rng=np.random.default_rng(seed))
        args.update(kwargs)
        return generate_phantom_cases_from_pool(**args)

    def test_phantoms_carry_drawn_covariates_and_model_p_value(self):
        phantoms, _ = self._generate()
        assert phantoms
        for p in phantoms:
            assert p.case_type == "Divorce and Separation"      # raw type, for eligibility
            assert p.court_type == "High Court"
            assert p.referral_mode == "Request by Parties"
            month_coef = 0.02 if p.referral_date.month == 3 else 0.0
            expected = 0.5 + month_coef + 0.05 + 0.04 + 0.01 - 0.1
            assert p.p_value == pytest.approx(expected)

    def test_discount_is_a_parameter(self):
        undiscounted, _ = self._generate(discount=0.0)
        discounted, _ = self._generate(discount=0.1)
        assert [p.p_value - 0.1 for p in undiscounted] == pytest.approx([p.p_value for p in discounted])

    def test_ids_negative_unique_and_next_id_returned(self):
        phantoms, next_id = self._generate(starting_id=-5)
        ids = [p.id for p in phantoms]
        assert all(i <= -5 for i in ids) and len(set(ids)) == len(ids)
        assert next_id == -5 - len(ids)

    def test_arrivals_within_horizon(self):
        phantoms, _ = self._generate()
        days = {(p.referral_date - date(2026, 3, 9)).days for p in phantoms}
        assert days <= set(range(10))

    def test_reproducible_with_same_rng_seed(self):
        a, _ = self._generate(seed=11)
        b, _ = self._generate(seed=11)
        assert [(p.id, p.referral_date, p.p_value) for p in a] == [(p.id, p.referral_date, p.p_value) for p in b]

    def test_mix_follows_pool_shares(self):
        pool = ArrivalPool(
            records=tuple(
                [{'case_type': "Civil Cases", 'court_station': "MILIMANI",
                  'referral_mode': "Referred by Court", 'court_type': "Magistrate Court"}] * 3
                + [{'case_type': "Civil Cases", 'court_station': "KAKAMEGA",
                    'referral_mode': "Referred by Court", 'court_type': "Magistrate Court"}]
            ),
            daily_rate=50.0,
        )
        phantoms, _ = self._generate(pool=pool, time_horizon=40)
        share = Counter(p.court_station for p in phantoms)["MILIMANI"] / len(phantoms)
        assert share == pytest.approx(0.75, abs=0.03)

    def test_empty_pool_gives_no_phantoms(self):
        phantoms, next_id = self._generate(pool=ArrivalPool(records=(), daily_rate=0.0))
        assert phantoms == [] and next_id == -1
