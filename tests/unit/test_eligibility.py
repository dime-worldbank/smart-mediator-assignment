import importlib.util
from datetime import date

import pytest

from smart_mediator_assignment.core import AssignmentIssue, MediatorProfile, MediatorRoster, SimpleCase
from smart_mediator_assignment.solver import LPSolver
from smart_mediator_assignment.solver.base import eligible_mediators_for_case
from tests.fixtures import SCENARIO1_VALID_MEDS, SCENARIO1_MED_BY_CRT_CASE_TYPE, SCENARIO1_MED_VA

_SOLVERS = [LPSolver]
if importlib.util.find_spec("osqp") is not None:
    from smart_mediator_assignment.solver import QPSolver
    _SOLVERS.append(QPSolver)


@pytest.mark.parametrize("solver_cls", _SOLVERS)
def test_per_case_eligibility(solver_cls):
    solver = solver_cls(
        valid_mediators=SCENARIO1_VALID_MEDS, mediator_case_loads={1: 0, 2: 0, 3: 0}, capacity=3,
        mediator_vas=SCENARIO1_MED_VA, med_by_court_case_type=SCENARIO1_MED_BY_CRT_CASE_TYPE,
        lambda_penalty=1.0, time_horizon=10,
    )

    def case(station="KAKAMEGA", eligible=None):
        return SimpleCase(id=1, case_type="Family group", court_station=station,
                          referral_date=date(2023, 1, 1), p_value=0.5, eligible_mediator_ids=eligible)

    def assigned(station="KAKAMEGA", eligible=None):
        result = solver.solve([case(station, eligible)], current_day=date(2023, 1, 1))
        return {m for m, p in result.get(1, []) if p > 1e-6}

    # unset: the court_station x case_type mapping, where mediator 1 has the best VA
    assert assigned() == {1}
    # set: replaces the mapping, e.g. mediator 1 already declined this case
    assert assigned(eligible=[2, 3]) and assigned(eligible=[2, 3]) <= {2, 3}
    # still restricted to the solver's valid mediators
    assert assigned(eligible=[99, 3]) == {3}
    # applies even where the mapping has no entry for the case
    assert assigned(station="UNMAPPED") == set()
    assert assigned(station="UNMAPPED", eligible=[2]) == {2}
    # a repeated id is one edge, not a duplicate the objective and capacity terms would double-count
    assert eligible_mediators_for_case(case(eligible=[3, 2, 3]), {}, {2, 3}) == [3, 2]


def test_roster_applies_the_production_rules():
    def mediator(mid, **overrides):
        base = dict(id=mid, is_active=True, court_stations=frozenset({"MOMBASA"}),
                    accreditation_categories=frozenset({"Family"}), religions=frozenset({"Islam"}))
        return MediatorProfile(**{**base, **overrides})

    roster = MediatorRoster(
        mediators=(
            mediator(1),
            mediator(2, is_active=False),
            mediator(3, pending_cases=3),
            mediator(4, court_stations=frozenset({"MILIMANI"})),
            mediator(5, accreditation_categories=frozenset({"Commercial"})),
            mediator(6, unavailable=((date(2023, 1, 1), date(2023, 1, 10)),)),
            mediator(7, religions=frozenset({"Christianity"})),
        ),
        case_type_accreditations={"Divorce and Separation": frozenset({"Family", "Children"})},
    )
    kadhi_case = dict(court_station="MOMBASA", case_type="Divorce and Separation", court_type="Kadhi Court")

    assert roster.assignment_issues(on_date=date(2023, 1, 9), max_caseload=3, **kadhi_case) == {
        1: [], 2: [AssignmentIssue.INACTIVE], 3: [AssignmentIssue.OVERCAPACITY],
        4: [AssignmentIssue.COURT_MISMATCH], 5: [AssignmentIssue.CASE_TYPE_MISMATCH],
        6: [AssignmentIssue.NOT_AVAILABLE], 7: [AssignmentIssue.RELIGION_MISMATCH],
    }
    # no cap (the smart algorithm), the unavailability end is exclusive, religion only for Kadhi courts
    assert roster.eligible_ids(on_date=date(2023, 1, 10), court_station="MOMBASA",
                               case_type="Divorce and Separation", court_type="Magistrate Court") == [1, 3, 6, 7]
    # unset case fields skip their checks, and exclude drops already-considered mediators
    assert roster.eligible_ids(on_date=date(2023, 2, 1), exclude=[1]) == [3, 4, 5, 6, 7]
