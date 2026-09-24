from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Collection, Dict, List, Optional, Tuple, Union

from ..core.types import MediatorId, CaseId, MedByCrtCaseType
from ..core.case import CaseProtocol


AssignmentDistribution = Dict[CaseId, List[Tuple[MediatorId, float]]]


def eligible_mediators_for_case(
    case: CaseProtocol,
    med_by_court_case_type: MedByCrtCaseType,
    valid_mediators: Collection[MediatorId],
) -> List[MediatorId]:
    """Mediators a case may be assigned to, restricted to `valid_mediators`.

    A case's own `eligible_mediator_ids` (when set) replaces the court-station x case-type
    lookup, so callers can apply rules the lookup can't express: per-case exclusions such
    as previously rejected mediators, Kadhi-court religion, unavailability on the date.
    """
    explicit = getattr(case, 'eligible_mediator_ids', None)
    if explicit is None:
        explicit = med_by_court_case_type.get(case.court_station, {}).get(case.case_type, [])
    # dict.fromkeys dedupes in order: a repeated id would add a duplicate edge to the LP/QP
    return [m for m in dict.fromkeys(explicit) if m in valid_mediators]


class BaseSolver(ABC):
    """Abstract base class for mediator assignment solvers."""

    @abstractmethod
    def solve(
        self,
        cases: List[CaseProtocol],
        phantom_cases: Optional[List[CaseProtocol]] = None,
        current_day: Optional[Union[date, datetime]] = None,
    ) -> AssignmentDistribution:
        """
        Solve the assignment problem.

        Args:
            cases: List of cases to assign
            phantom_cases: Optional list of phantom (future) cases
            current_day: Current date for time-horizon calculations

        Returns:
            Dictionary mapping case_id -> [(mediator_id, probability), ...]
            sorted by probability descending
        """
        pass
