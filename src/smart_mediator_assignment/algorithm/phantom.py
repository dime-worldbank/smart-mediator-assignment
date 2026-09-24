"""
Phantom case generation for look-ahead optimization.

This module generates simulated future cases to help the solver make forward-looking
assignment decisions. `generate_phantom_cases_from_pool` draws each phantom case's full
covariate vector from recent real arrivals and scores it with the fitted VA model;
`generate_phantom_cases` is the earlier per-cell Poisson sampler with averaged p-values.
"""

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

from ..core.case import CaseProtocol, SimpleCase
from .va_estimation import VAModel
from ..core.types import (
    AvgCaseRate,
    AvgPValByCrtCaseType,
    MedByCrtCaseType,
    CourtStationId,
    CaseTypeId,
)


def generate_phantom_cases(
    current_day: Union[date, datetime],
    time_horizon: int,
    avg_case_rate: AvgCaseRate,
    avg_p_val_by_crt_case_type: AvgPValByCrtCaseType,
    med_by_court_case_type: MedByCrtCaseType,
    court_stations: List[CourtStationId],
    case_types: List[CaseTypeId],
    starting_id: int = -1,
    default_p_val: float = 0.5,
    seed: int = None,
) -> Tuple[List[SimpleCase], int]:
    """
    Generate phantom (simulated future) cases using Poisson sampling.

    Args:
        current_day: Current date
        time_horizon: Number of days to look ahead
        avg_case_rate: Average daily case arrival rates by case_type and court_station
        avg_p_val_by_crt_case_type: Average p-values by (case_type, court_station)
        med_by_court_case_type: Mapping of eligible mediators
        court_stations: List of court station IDs to consider
        case_types: List of case type IDs to consider
        starting_id: Starting ID for phantom cases (should be negative)
        default_p_val: Default p-value when not found in mapping
        seed: Random seed for reproducibility

    Returns:
        Tuple of (list of phantom cases, next available phantom ID)
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    phantom_id = starting_id
    phantom_cases_with_order = []

    for fut_day in range(time_horizon):
        arrival_date = _add_days(current_day, fut_day)

        pairs = [
            (court_station, case_type)
            for court_station in court_stations
            for case_type in case_types
        ]

        for court_station, case_type in pairs:
            if court_station not in med_by_court_case_type:
                continue
            if case_type not in med_by_court_case_type[court_station]:
                continue

            if case_type not in avg_case_rate:
                continue
            if court_station not in avg_case_rate[case_type]:
                continue

            lambda_rate = avg_case_rate[case_type][court_station]
            num_cases = rng.poisson(lambda_rate)

            for _ in range(num_cases):
                p_val_key = (case_type, court_station)
                if p_val_key in avg_p_val_by_crt_case_type:
                    p_val = avg_p_val_by_crt_case_type[p_val_key]
                else:
                    p_val = default_p_val

                order_key = py_rng.uniform(0, 1)

                phantom_case = SimpleCase(
                    id=phantom_id,
                    case_type=case_type,
                    court_station=court_station,
                    referral_date=arrival_date,
                    p_value=p_val - 0.1,
                )

                phantom_cases_with_order.append((order_key, phantom_case))
                phantom_id -= 1

    phantom_cases_with_order.sort(key=lambda x: x[0])
    phantom_cases = [case for _, case in phantom_cases_with_order]

    return phantom_cases, phantom_id


@dataclass(frozen=True)
class ArrivalPool:
    """Covariate vectors of recent real arrivals, and their daily arrival rate."""

    records: Tuple[Dict[str, str], ...]
    daily_rate: float


def build_arrival_pool(
    cases: Iterable[CaseProtocol],
    as_of: Union[date, datetime],
    window_days: int = 182,
) -> ArrivalPool:
    """Pool of cases referred in the `window_days` up to and including `as_of`.

    Arrival is the referral date. Each record keeps the raw case type (eligibility is
    keyed on it) and court type (Kadhi-court eligibility, high court / court of appeal
    covariates) alongside the other covariates the VA model uses.
    """
    end = pd.Timestamp(as_of).normalize()
    start = end - pd.Timedelta(days=window_days)
    records = []
    for case in cases:
        if case.referral_date is None or pd.isna(case.referral_date):
            continue
        referred = pd.Timestamp(case.referral_date).normalize()
        if start < referred <= end:
            records.append({
                'case_type': case.case_type,
                'court_station': case.court_station,
                'referral_mode': getattr(case, 'referral_mode', ''),
                'court_type': getattr(case, 'court_type', ''),
            })
    return ArrivalPool(records=tuple(records), daily_rate=len(records) / window_days)


def generate_phantom_cases_from_pool(
    current_day: Union[date, datetime],
    time_horizon: int,
    pool: ArrivalPool,
    va_model: VAModel,
    rng: np.random.Generator,
    starting_id: int = -1,
    discount: float = 0.1,
) -> Tuple[List[SimpleCase], int]:
    """Phantom cases for the next `time_horizon` days, drawn from recent arrivals.

    Each day draws a Poisson(`pool.daily_rate`) count, then that many covariate vectors
    uniformly from the pool, so the mix across court stations, case types and other
    covariates matches recent arrivals. Each phantom's p_value is the VA model's
    prediction at its arrival date, minus `discount`.

    Returns:
        Tuple of (list of phantom cases, next available phantom ID)
    """
    phantom_id = starting_id
    phantom_cases = []
    if not pool.records:
        return phantom_cases, phantom_id

    for fut_day in range(time_horizon):
        arrival_date = _add_days(current_day, fut_day)
        count = rng.poisson(pool.daily_rate)
        for idx in rng.integers(0, len(pool.records), size=count):
            record = pool.records[idx]
            p_val = va_model.predict_arrival(arrival_date=arrival_date, **record) - discount
            phantom_cases.append(SimpleCase(
                id=phantom_id,
                case_type=record['case_type'],
                court_station=record['court_station'],
                referral_date=arrival_date,
                p_value=p_val,
                court_type=record['court_type'],
                referral_mode=record['referral_mode'],
            ))
            phantom_id -= 1

    # Interleave days and cells like generate_phantom_cases does, so solver input order
    # doesn't encode arrival day.
    order = rng.permutation(len(phantom_cases))
    return [phantom_cases[i] for i in order], phantom_id


def _add_days(
    base_date: Union[date, datetime], days: int
) -> Union[date, datetime]:
    """Add days to a date or datetime object."""
    if isinstance(base_date, datetime):
        return base_date + timedelta(days=days)
    return base_date + timedelta(days=days)


def estimate_case_arrivals(
    avg_case_rate: AvgCaseRate,
    court_stations: List[CourtStationId],
    case_types: List[CaseTypeId],
    days: int = 1,
) -> Dict[Tuple[CaseTypeId, CourtStationId], float]:
    """
    Estimate expected case arrivals over a period.

    Args:
        avg_case_rate: Average daily case rates
        court_stations: Court stations to consider
        case_types: Case types to consider
        days: Number of days to estimate over

    Returns:
        Dictionary mapping (case_type, court_station) -> expected arrivals
    """
    estimates = {}

    for case_type in case_types:
        if case_type not in avg_case_rate:
            continue

        for court_station in court_stations:
            if court_station not in avg_case_rate[case_type]:
                continue

            lambda_rate = avg_case_rate[case_type][court_station]
            estimates[(case_type, court_station)] = lambda_rate * days

    return estimates
