"""Arms for court station x case type cells missing from the RCT's randomized mapping.

Such cells (a new court station, a case type a station never had, a cell the clustering left
out) are given an arm so their cases can be assigned, but they are not part of the experiment.
Agreed by the research team: at the start, each mediator's Tbar is the share of their earlier
cases that were in Treatment cells; a new cell takes Treatment when its mediators' average Tbar
is above 0.5, and the smart algorithm by default when the average is exactly 0.5 or the cell
has no mediators; otherwise Control.
"""
from collections import Counter
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..core.types import MediatorId

TREATMENT = 'Treatment'
CONTROL = 'Control'
# share for a mediator with no earlier case in a Treatment or Control cell
NEW_MEDIATOR_SHARE = 0.5


def mediator_treatment_shares(
    case_arms: Iterable[Tuple[MediatorId, Optional[str]]],
) -> Dict[MediatorId, Tuple[float, int]]:
    """Each mediator's Tbar and the number of cases it is based on, from (mediator id, arm of
    the case's cell) pairs. A case whose cell has no arm (None) is left out; a mediator with no
    counted case is absent and so takes NEW_MEDIATOR_SHARE."""
    treated: Counter = Counter()
    total: Counter = Counter()
    for mediator_id, arm in case_arms:
        if arm is None:
            continue
        if arm not in (TREATMENT, CONTROL):
            raise ValueError(f"arm must be {TREATMENT!r}, {CONTROL!r} or None, got {arm!r}")
        treated[mediator_id] += arm == TREATMENT
        total[mediator_id] += 1
    return {m: (treated[m] / n, n) for m, n in total.items()}


def classify_unmapped_cell(
    mediator_ids: Sequence[MediatorId],
    shares: Mapping[MediatorId, float],
) -> str:
    """The arm a new cell is treated as, from the Tbar of the mediators who can take its cases."""
    if not mediator_ids:
        return TREATMENT
    average = sum(shares.get(m, NEW_MEDIATOR_SHARE) for m in mediator_ids) / len(mediator_ids)
    return TREATMENT if average >= NEW_MEDIATOR_SHARE else CONTROL
