"""The belief rules the RCT simulation applies (cadaster-algo RCTSimulation/simulation_RCT.py:482-510),
shared by the production app so both keep mediator beliefs the same way."""
import json
from datetime import date, datetime

import pytest

from smart_mediator_assignment import (
    BeliefEngine,
    MediatorVAEstimate,
    VAEstimationResult,
    compute_posterior,
    estimate_va,
    estimate_va_as_of,
    initial_mediator_sds,
    VAEstimationConfig,
)
from smart_mediator_assignment.core.case import SimpleCase


def fake_va(vas, sigma=0.3):
    return VAEstimationResult(
        mediator_vas=[MediatorVAEstimate(mediator_id=m, va=v, n_cases=5) for m, v in vas.items()],
        case_predictions=[],
        sigma=sigma,
    )


def started(vas, sds, mediator_ids, global_sigma=0.3):
    return BeliefEngine.start(
        va=fake_va(vas, sigma=global_sigma), sds=sds, mediator_ids=mediator_ids, global_sigma=global_sigma
    )


def test_start_takes_va_as_mean_and_given_sd_as_prior():
    engine = started({1: 0.2}, {1: 0.1}, [1])
    assert (engine.mu[1], engine.sd[1]) == (0.2, 0.1)
    assert (engine.prior_mu[1], engine.prior_sd[1]) == (0.2, 0.1)
    assert engine.history[1] == ([], [])


def test_mediator_missing_from_va_starts_at_global_prior():
    # run_RCT_sim.py:227 — a mediator with no VA estimate starts at (0, sd_global)
    engine = started({1: 0.2}, {1: 0.1, 9: 0.05}, [1, 9], global_sigma=0.3)
    assert (engine.mu[9], engine.sd[9]) == (0.0, 0.3)


def test_record_conclusion_files_p_under_its_outcome():
    engine = started({1: 0.2}, {1: 0.1}, [1])
    engine.record_conclusion(1, outcome=0, p_value=0.3)
    engine.record_conclusion(1, outcome=1, p_value=0.4)
    assert engine.history[1] == ([0.3], [0.4])


def test_record_conclusion_rejects_non_binary_outcome():
    engine = started({1: 0.2}, {1: 0.1}, [1])
    with pytest.raises(ValueError):
        engine.record_conclusion(1, outcome=2, p_value=0.4)


def test_daily_update_is_posterior_from_prior_with_history():
    engine = started({1: 0.2}, {1: 0.1}, [1])
    engine.record_conclusion(1, outcome=1, p_value=0.4)
    engine.daily_update()
    assert (engine.mu[1], engine.sd[1]) == compute_posterior(
        ([], [0.4]), prior_mu=0.2, prior_sigma=0.1, prev_mu=0.2, prev_sigma=0.1)


def test_daily_update_keeps_history_until_refit():
    engine = started({1: 0.2}, {1: 0.1}, [1])
    engine.record_conclusion(1, outcome=1, p_value=0.4)
    engine.daily_update()
    assert engine.history[1] == ([], [0.4])


def test_refit_takes_new_va_as_mean_posterior_sd_and_resets_history():
    engine = started({1: 0.2, 2: -0.1}, {1: 0.1, 2: 0.2}, [1, 2])
    engine.record_conclusion(1, outcome=1, p_value=0.4)
    engine.record_conclusion(2, outcome=0, p_value=0.6)
    engine.daily_update()
    mu1, sd1 = engine.mu[1], engine.sd[1]
    mu2_before, sd2_before, history2_before = engine.mu[2], engine.sd[2], engine.history[2]
    _, expected_sd = compute_posterior(
        ([], [0.4]), prior_mu=0.2, prior_sigma=0.1, prev_mu=mu1, prev_sigma=sd1)

    engine.refit(fake_va({1: 0.5}))  # mediator 2 absent from the re-fit

    assert engine.mu[1] == 0.5 and engine.prior_mu[1] == 0.5
    assert engine.sd[1] == expected_sd and engine.prior_sd[1] == expected_sd
    assert engine.history[1] == ([], [])
    assert (engine.mu[2], engine.sd[2], engine.history[2]) == (mu2_before, sd2_before, history2_before)


def test_engine_round_trips_through_json():
    engine = started({1: 0.2, 2: -0.1}, {1: 0.1, 2: 0.2}, [1, 2])
    engine.record_conclusion(1, outcome=1, p_value=0.4)
    engine.daily_update()
    assert BeliefEngine.from_dict(json.loads(json.dumps(engine.to_dict()))) == engine


def _case(cid, med, outcome, referral, appt, concl, status="CONCLUDED"):
    return SimpleCase(
        id=cid, case_type="Family group", court_station="MILIMANI",
        referral_date=referral, mediator_id=med, case_outcome_agreement=outcome,
        mediator_appointment_date=appt, conclusion_date=concl, case_status=status,
        court_type="Magistrate", referral_mode="Referred by Court",
    )


def _va_with_predictions(preds, sigma=0.3):
    from smart_mediator_assignment import CasePrediction
    return VAEstimationResult(
        mediator_vas=[], sigma=sigma,
        case_predictions=[CasePrediction(case_id=c, mediator_id=0, p_pred=p, va=0.0,
                                         case_outcome_agreement=None) for c, p in preds.items()],
    )


def test_initial_sd_is_posterior_over_cases_concluded_before_as_of():
    # simulation_utils.estimate_med_sds: prior (0, global sigma), cases referred and
    # concluded before the start, only when a mediator has more than one
    as_of = date(2025, 9, 1)
    cases = [
        _case(1, 7, 1, date(2025, 1, 1), date(2025, 1, 5), date(2025, 3, 1)),
        _case(2, 7, 0, date(2025, 2, 1), date(2025, 2, 5), date(2025, 4, 1)),
        _case(3, 7, 1, date(2025, 2, 1), date(2025, 2, 5), date(2025, 10, 1)),  # concluded after as_of
    ]
    va = _va_with_predictions({1: 0.4, 2: 0.6, 3: 0.5})
    sds = initial_mediator_sds(cases, va, [7], as_of)
    assert sds[7] == compute_posterior(([0.6], [0.4]), prior_mu=0, prior_sigma=0.3)[1]


def test_initial_sd_is_global_sigma_with_one_case_or_fewer():
    as_of = date(2025, 9, 1)
    cases = [_case(1, 7, 1, date(2025, 1, 1), date(2025, 1, 5), date(2025, 3, 1))]
    sds = initial_mediator_sds(cases, _va_with_predictions({1: 0.4}), [7, 8], as_of)
    assert sds == {7: 0.3, 8: 0.3}


def test_initial_sd_skips_cases_without_a_prediction():
    as_of = date(2025, 9, 1)
    cases = [
        _case(1, 7, 1, date(2025, 1, 1), date(2025, 1, 5), date(2025, 3, 1)),
        _case(2, 7, 0, date(2025, 2, 1), date(2025, 2, 5), date(2025, 4, 1)),
    ]
    sds = initial_mediator_sds(cases, _va_with_predictions({1: 0.4}), [7], as_of)
    assert sds[7] == 0.3  # one usable case left


def test_estimate_va_as_of_is_the_fit_as_of_that_date():
    cases = [
        _case(i, (i % 3) + 1, i % 2, date(2022, 1, 1), date(2022, 1, 10), date(2022, 2, 1))
        for i in range(12)
    ]
    as_of = datetime(2023, 6, 1)
    expected = estimate_va(cases, VAEstimationConfig(reference_date=as_of),
                           start_date=datetime(2016, 4, 6), end_date=as_of)
    result = estimate_va_as_of(cases, as_of=as_of.date(), start_date=date(2016, 4, 6))
    assert result.get_va_dict() == expected.get_va_dict()
    assert result.sigma == expected.sigma


def test_estimate_va_as_of_ignores_cases_referred_after_as_of():
    base = [
        _case(i, (i % 3) + 1, i % 2, date(2022, 1, 1), date(2022, 1, 10), date(2022, 2, 1))
        for i in range(12)
    ]
    later = _case(99, 1, 1, date(2023, 7, 1), date(2023, 7, 2), date(2023, 7, 3))
    as_of = date(2023, 6, 1)
    with_later = estimate_va_as_of(base + [later], as_of, date(2016, 4, 6))
    assert 99 not in with_later.get_p_pred_dict()
    assert with_later.get_va_dict() == estimate_va_as_of(base, as_of, date(2016, 4, 6)).get_va_dict()


def test_estimate_va_as_of_ignores_cases_appointed_before_start_date():
    base = [
        _case(i, (i % 3) + 1, i % 2, date(2022, 1, 1), date(2022, 1, 10), date(2022, 2, 1))
        for i in range(12)
    ]
    early = [
        _case(100 + i, 1, 1, date(2015, 1, 1), date(2015, 1, 10), date(2015, 2, 1))
        for i in range(6)
    ]
    as_of = date(2023, 6, 1)
    with_early = estimate_va_as_of(base + early, as_of, date(2016, 4, 6))
    assert not {100 + i for i in range(6)} & set(with_early.get_p_pred_dict())
    assert with_early.get_va_dict() == estimate_va_as_of(base, as_of, date(2016, 4, 6)).get_va_dict()


def test_estimate_va_as_of_keeps_unappointed_cases_for_prediction():
    base = [
        _case(i, (i % 3) + 1, i % 2, date(2022, 1, 1), date(2022, 1, 10), date(2022, 2, 1))
        for i in range(12)
    ]
    waiting = _case(200, None, None, date(2023, 5, 1), None, None, status="PENDING")
    result = estimate_va_as_of(base + [waiting], date(2023, 6, 1), date(2016, 4, 6))
    assert 200 in result.get_p_pred_dict()
