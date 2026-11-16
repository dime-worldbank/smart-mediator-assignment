"""Mediator beliefs over the course of the RCT, as the research simulation keeps them.

Ported from cadaster-algo (RCTSimulation/simulation_RCT.py:482-510, run_RCT_sim.py:208-227,
simulation_utils.py::estimate_med_sds) so the production app and the simulation share one
implementation: a start state from a VA fit, a posterior update on ordinary days, and a VA
re-fit that resets each mediator's prior on re-fit days. Callers decide which day is which.
"""
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Tuple, Union

from ..core.case import CaseProtocol
from ..core.types import MediatorId
from .bayesian import compute_posterior
from .va_estimation import VAEstimationConfig, VAEstimationResult, estimate_va


def _as_datetime(day: Union[date, datetime]) -> datetime:
    if isinstance(day, datetime):
        return day
    return datetime(day.year, day.month, day.day)


def estimate_va_as_of(
    cases: List[CaseProtocol],
    as_of: Union[date, datetime],
    start_date: Union[date, datetime],
) -> VAEstimationResult:
    """The VA fit as it would run on `as_of`.

    `cases` must reflect what was known on `as_of`: a case concluded later is still PENDING
    with no conclusion date or outcome. Cases referred after `as_of`, and cases appointed
    before `start_date`, are ignored; cases not yet appointed are kept for prediction. Case
    ages (the 180-day and pending rules) are measured from `as_of`.
    """
    as_of_dt = _as_datetime(as_of)
    start_dt = _as_datetime(start_date)
    # estimate_va applies its own start_date window only when end_date precedes the
    # reference date, which is never the case here, so the start is enforced on the input
    known = [
        c for c in cases
        if c.referral_date is not None and _as_datetime(c.referral_date) <= as_of_dt
        and (c.mediator_appointment_date is None or _as_datetime(c.mediator_appointment_date) >= start_dt)
    ]
    return estimate_va(
        known,
        VAEstimationConfig(reference_date=as_of_dt),
        start_date=start_dt,
        end_date=as_of_dt,
    )


def initial_mediator_sds(
    cases: Iterable[CaseProtocol],
    va: VAEstimationResult,
    mediator_ids: Iterable[MediatorId],
    as_of: Union[date, datetime],
) -> Dict[MediatorId, float]:
    """Each mediator's starting belief sd: the posterior sd over their cases referred and
    concluded before `as_of`, from a (0, va.sigma) prior; va.sigma when they have one such
    case or none. A case's p is its prediction in `va`; cases without one are skipped."""
    as_of_dt = _as_datetime(as_of)
    p_pred = va.get_p_pred_dict()
    wanted = set(mediator_ids)
    history: Dict[MediatorId, Tuple[List[float], List[float]]] = {m: ([], []) for m in wanted}
    for case in cases:
        if case.mediator_id not in wanted:
            continue
        if case.referral_date is None or case.conclusion_date is None \
                or case.mediator_appointment_date is None:
            continue
        if not (_as_datetime(case.referral_date) < as_of_dt
                and _as_datetime(case.conclusion_date) < as_of_dt):
            continue
        p = p_pred.get(case.id)
        if p is None or math.isnan(p):
            continue
        outcome = case.case_outcome_agreement
        if outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
            continue
        if outcome not in (0, 1):
            raise ValueError(f"Unexpected case_outcome_agreement for case {case.id}: {outcome!r}")
        history[case.mediator_id][int(outcome)].append(p)

    sds = {}
    for mediator_id in wanted:
        failed, succeeded = history[mediator_id]
        if len(failed) + len(succeeded) > 1:
            _, sds[mediator_id] = compute_posterior(
                (failed, succeeded), prior_mu=0, prior_sigma=va.sigma)
        else:
            sds[mediator_id] = va.sigma
    return sds


@dataclass
class BeliefEngine:
    """Per-mediator belief mean and sd, the prior they are updated from, and the outcomes
    (as the p each case was assigned with) seen since the last re-fit."""

    mu: Dict[MediatorId, float]
    sd: Dict[MediatorId, float]
    prior_mu: Dict[MediatorId, float]
    prior_sd: Dict[MediatorId, float]
    history: Dict[MediatorId, Tuple[List[float], List[float]]]
    global_sigma: float

    @classmethod
    def start(
        cls,
        va: VAEstimationResult,
        sds: Dict[MediatorId, float],
        mediator_ids: Iterable[MediatorId],
        global_sigma: float,
    ) -> "BeliefEngine":
        """A mediator in `va` starts at (their VA, sds[mediator]); one missing from it
        starts at (0, global_sigma)."""
        vas = va.get_va_dict()
        mu, sd = {}, {}
        for mediator_id in mediator_ids:
            if mediator_id in vas:
                mu[mediator_id], sd[mediator_id] = vas[mediator_id], sds[mediator_id]
            else:
                mu[mediator_id], sd[mediator_id] = 0.0, global_sigma
        return cls(
            mu=mu, sd=sd, prior_mu=dict(mu), prior_sd=dict(sd),
            history={m: ([], []) for m in mu}, global_sigma=global_sigma,
        )

    def record_conclusion(self, mediator_id: MediatorId, outcome: int, p_value: float) -> None:
        if outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
        self.history[mediator_id][outcome].append(p_value)

    def daily_update(self) -> None:
        for mediator_id in self.mu:
            self.mu[mediator_id], self.sd[mediator_id] = compute_posterior(
                self.history[mediator_id],
                prior_mu=self.prior_mu[mediator_id],
                prior_sigma=self.prior_sd[mediator_id],
                prev_mu=self.mu[mediator_id],
                prev_sigma=self.sd[mediator_id],
            )

    def refit(self, va: VAEstimationResult) -> None:
        """Mediators in the re-fit take its VA as their mean and the posterior sd over their
        history; both become their new prior and their history is cleared. Mediators missing
        from the re-fit are left as they were."""
        vas = va.get_va_dict()
        for mediator_id in self.mu:
            if mediator_id not in vas:
                continue
            _, self.sd[mediator_id] = compute_posterior(
                self.history[mediator_id],
                prior_mu=self.prior_mu[mediator_id],
                prior_sigma=self.prior_sd[mediator_id],
                prev_mu=self.mu[mediator_id],
                prev_sigma=self.sd[mediator_id],
            )
            self.mu[mediator_id] = vas[mediator_id]
            self.prior_mu[mediator_id] = self.mu[mediator_id]
            self.prior_sd[mediator_id] = self.sd[mediator_id]
            self.history[mediator_id] = ([], [])

    def to_dict(self) -> dict:
        """JSON-safe form; mediator ids become strings, as JSON object keys must."""
        return {
            'global_sigma': self.global_sigma,
            'mediators': {
                str(m): {
                    'mu': self.mu[m], 'sd': self.sd[m],
                    'prior_mu': self.prior_mu[m], 'prior_sd': self.prior_sd[m],
                    'failed': list(self.history[m][0]), 'succeeded': list(self.history[m][1]),
                }
                for m in self.mu
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BeliefEngine":
        mediators = {int(m): v for m, v in data['mediators'].items()}
        return cls(
            mu={m: v['mu'] for m, v in mediators.items()},
            sd={m: v['sd'] for m, v in mediators.items()},
            prior_mu={m: v['prior_mu'] for m, v in mediators.items()},
            prior_sd={m: v['prior_sd'] for m, v in mediators.items()},
            history={m: (list(v['failed']), list(v['succeeded'])) for m, v in mediators.items()},
            global_sigma=data['global_sigma'],
        )
