import importlib.util
from datetime import date

import pytest

from smart_mediator_assignment.core import SimpleCase
from smart_mediator_assignment.solver import LPSolver
from smart_mediator_assignment.solver.base import eligible_mediators_for_case
from tests.fixtures import (
    SCENARIO1_VALID_MEDS,
    SCENARIO1_MED_BY_CRT_CASE_TYPE,
    SCENARIO1_MED_VA,
)

osqp_missing = importlib.util.find_spec("osqp") is None


def _case(station="KAKAMEGA", eligible=None, case_id=1):
    return SimpleCase(id=case_id, case_type="Family group", court_station=station,
                      referral_date=date(2023, 1, 1), p_value=0.5, eligible_mediator_ids=eligible)


def test_mapping_used_when_no_explicit_list():
    assert eligible_mediators_for_case(_case(), SCENARIO1_MED_BY_CRT_CASE_TYPE, {1, 2, 3}) == [1, 2, 3]


def test_explicit_list_replaces_mapping():
    assert eligible_mediators_for_case(_case(eligible=[3]), SCENARIO1_MED_BY_CRT_CASE_TYPE, {1, 2, 3}) == [3]


def test_explicit_list_still_restricted_to_valid_mediators():
    # an explicit list can't reintroduce a mediator the solver wasn't given
    assert eligible_mediators_for_case(_case(eligible=[2, 99]), SCENARIO1_MED_BY_CRT_CASE_TYPE, {1, 2, 3}) == [2]


def test_explicit_list_applies_even_without_a_mapping_entry():
    assert eligible_mediators_for_case(_case(station="UNMAPPED", eligible=[1]),
                                       SCENARIO1_MED_BY_CRT_CASE_TYPE, {1, 2, 3}) == [1]
    assert eligible_mediators_for_case(_case(station="UNMAPPED"),
                                       SCENARIO1_MED_BY_CRT_CASE_TYPE, {1, 2, 3}) == []


def test_empty_explicit_list_means_no_eligible_mediator():
    assert eligible_mediators_for_case(_case(eligible=[]), SCENARIO1_MED_BY_CRT_CASE_TYPE, {1, 2, 3}) == []


def _solvers():
    solvers = [LPSolver]
    if not osqp_missing:
        from smart_mediator_assignment.solver import QPSolver
        solvers.append(QPSolver)
    return solvers


@pytest.mark.parametrize("solver_cls", _solvers())
def test_solver_honors_per_case_exclusion(solver_cls):
    # Mediator 1 has the highest VA at KAKAMEGA; excluding it for this case (e.g. it was
    # already rejected) must route the case to the remaining mediators only.
    solver = solver_cls(
        valid_mediators=SCENARIO1_VALID_MEDS,
        mediator_case_loads={1: 0, 2: 0, 3: 0},
        capacity=3,
        mediator_vas=SCENARIO1_MED_VA,
        med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
        lambda_penalty=1.0,
        time_horizon=10,
    )
    unrestricted = solver.solve([_case()], current_day=date(2023, 1, 1))
    assert unrestricted[1][0][0] == 1

    restricted = solver.solve([_case(eligible=[2, 3])], current_day=date(2023, 1, 1))
    assigned = {m for m, p in restricted[1] if p > 1e-6}
    assert assigned and assigned <= {2, 3}
