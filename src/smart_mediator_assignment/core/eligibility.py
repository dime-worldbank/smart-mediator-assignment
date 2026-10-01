"""Mediator eligibility rules, shared by the production app and the simulation.

Mirrors cadaster-kenya-mediation's `get_mediator_assignment_issues`, over plain mediator
snapshots instead of database queries, so callers build a `MediatorRoster` from their own data.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Collection, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

from .types import CaseTypeId, CourtStationId, MediatorId

KADHI_COURT = 'Kadhi Court'
ISLAM = 'Islam'


class AssignmentIssue(str, Enum):
    INACTIVE = 'inactive'
    OVERCAPACITY = 'overcapacity'
    COURT_MISMATCH = 'court_mismatch'
    CASE_TYPE_MISMATCH = 'case_type_mismatch'
    NOT_AVAILABLE = 'not_available'
    RELIGION_MISMATCH = 'religion_mismatch'


@dataclass(frozen=True)
class MediatorProfile:
    """What the eligibility rules need to know about a mediator."""

    id: MediatorId
    is_active: bool
    court_stations: FrozenSet[CourtStationId]
    accreditation_categories: FrozenSet[str]
    religions: FrozenSet[str] = frozenset()
    # (start, end) with end exclusive, like a Postgres daterange; None is unbounded.
    unavailable: Tuple[Tuple[Optional[date], Optional[date]], ...] = ()
    pending_cases: int = 0

    def is_unavailable_on(self, day: date) -> bool:
        return any((start is None or start <= day) and (end is None or day < end)
                   for start, end in self.unavailable)


@dataclass(frozen=True)
class MediatorRoster:
    """Mediators and the accreditation categories each case type accepts.

    Keys must match the cases': court stations as in `case.court_station`, case types as in
    `case.case_type`.
    """

    mediators: Tuple[MediatorProfile, ...]
    case_type_accreditations: Mapping[CaseTypeId, FrozenSet[str]]

    def assignment_issues(
        self,
        *,
        on_date: Union[date, datetime],
        court_station: Optional[CourtStationId] = None,
        case_type: Optional[CaseTypeId] = None,
        court_type: Optional[str] = None,
        max_caseload: Optional[int] = None,
    ) -> Dict[MediatorId, List[AssignmentIssue]]:
        """Each mediator's reasons it can't take the case; none means eligible.

        A case field left as None skips its check, as in production. `max_caseload=None` applies
        no cap: production's Control arm caps pending cases at 3, the smart algorithm does not.
        """
        day = on_date.date() if isinstance(on_date, datetime) else on_date
        accepted = self.case_type_accreditations.get(case_type, frozenset())
        issues = {}
        for mediator in self.mediators:
            found = []
            if not mediator.is_active:
                found.append(AssignmentIssue.INACTIVE)
            if max_caseload is not None and mediator.pending_cases >= max_caseload:
                found.append(AssignmentIssue.OVERCAPACITY)
            if court_station is not None and court_station not in mediator.court_stations:
                found.append(AssignmentIssue.COURT_MISMATCH)
            if case_type is not None and not (mediator.accreditation_categories & accepted):
                found.append(AssignmentIssue.CASE_TYPE_MISMATCH)
            if mediator.is_unavailable_on(day):
                found.append(AssignmentIssue.NOT_AVAILABLE)
            if court_type == KADHI_COURT and ISLAM not in mediator.religions:
                found.append(AssignmentIssue.RELIGION_MISMATCH)
            issues[mediator.id] = found
        return issues

    def eligible_ids(
        self,
        *,
        on_date: Union[date, datetime],
        court_station: Optional[CourtStationId] = None,
        case_type: Optional[CaseTypeId] = None,
        court_type: Optional[str] = None,
        max_caseload: Optional[int] = None,
        exclude: Collection[MediatorId] = (),
    ) -> List[MediatorId]:
        """Mediators with no assignment issue, minus `exclude` (e.g. already recommended or
        declined for this case), in roster order."""
        issues = self.assignment_issues(on_date=on_date, court_station=court_station, case_type=case_type,
                                        court_type=court_type, max_caseload=max_caseload)
        return [m for m, found in issues.items() if not found and m not in exclude]
