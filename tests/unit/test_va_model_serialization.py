import json
import math
from datetime import date, datetime

from smart_mediator_assignment import VAModel


def _model():
    return VAModel(
        intercept=0.49,
        coefficients={
            'casetype_simplified': {'Civil group': 0.03, 'zzzSmall': float('nan')},
            'court_station': {'MOMBASA': -0.02},
            'appt_month': {3.0: 0.01, 11.0: -0.04},
            'quasiyear': {1.0: 0.02, 2.0: -0.01},
            'highcourt': {1.0: 0.05},
        },
        small_court_stations=frozenset({'KOMBEWA', 'OLOKURTO'}),
        small_case_types=frozenset({'Constitution and Human Rights'}),
        quasiyear_anchor=datetime(2025, 9, 1),
        merged_quasiyear=9,
    )


COVARIATES = dict(case_type='Civil Cases', court_station='MOMBASA',
                  referral_mode='Referred by Court', court_type='High Court')


def _round_trip(model):
    # allow_nan=False: Postgres JSON rejects NaN, so the dict must not contain any
    return VAModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))


def test_va_model_round_trip_predicts_identically():
    model = _model()
    restored = _round_trip(model)
    for arrival in (date(2025, 3, 2), date(2025, 11, 20), date(2025, 6, 1)):
        assert restored.predict_arrival(arrival_date=arrival, **COVARIATES) == \
            model.predict_arrival(arrival_date=arrival, **COVARIATES)


def test_va_model_round_trip_keeps_numeric_levels_and_metadata():
    restored = _round_trip(_model())
    assert restored.coefficients['appt_month'] == {3.0: 0.01, 11.0: -0.04}
    assert restored.small_court_stations == frozenset({'KOMBEWA', 'OLOKURTO'})
    assert restored.quasiyear_anchor == datetime(2025, 9, 1)
    assert restored.merged_quasiyear == 9
    assert restored.quasiyear_of(date(2024, 8, 15)) == _model().quasiyear_of(date(2024, 8, 15))


def test_nan_coefficient_survives_as_a_skipped_level():
    restored = _round_trip(_model())
    assert math.isnan(restored.coefficients['casetype_simplified']['zzzSmall'])
