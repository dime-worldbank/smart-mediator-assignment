# Smart Mediator Assignment

An LP-based algorithm for optimizing mediator assignments in court mediation systems.

## Installation

Using uv (recommended):
```bash
uv sync
source .venv/bin/activate
```

Optional variants:

```bash
uv sync --dev                  # development dependencies
uv sync --extra gurobi         # Gurobi support
```

Using pip (alternative):
```bash
pip install -e .
pip install -e ".[dev]"  # for development
pip install -e ".[gurobi]"  # for Gurobi support
```

## Usage

### Basic Usage

```python
from datetime import date
from smart_mediator_assignment import (
    SimpleCase,
    BeliefState,
    AlgorithmConfig,
    get_recommendations,
)

# Create a case
case = SimpleCase(
    id=1,
    case_type_id="Family group",
    court_station_id="KAKAMEGA",
    referral_date=date(2023, 1, 1),
    p_value=0.5,
)

# Set up belief state
belief_state = BeliefState.from_init_va(
    {1: {"mu": 0.0, "sd": 0.12}, 2: {"mu": 0.05, "sd": 0.12}},
    global_sigma=0.12,
)

# Configure algorithm
config = AlgorithmConfig(capacity=3, lambda_penalty=1.0, time_horizon=10)

# Get recommendations
result = get_recommendations(
    case=case,
    eligible_mediator_ids=[1, 2],
    mediator_case_loads={1: 0, 2: 1},
    belief_state=belief_state,
    med_by_court_case_type={"KAKAMEGA": {"Family group": [1, 2]}},
    config=config,
    current_day=date(2023, 1, 1),
)

top_mediator = result.get_top_mediator()
```

### Belief Updates

```python
from smart_mediator_assignment import update_belief

# After case resolution
new_belief_state = update_belief(
    belief_state=belief_state,
    mediator_id=1,
    case_p_val=0.5,
    outcome=1,  # 1 for success, 0 for failure
)
```

### Batch Recommendations

```python
from smart_mediator_assignment import get_recommendations_batch

results = get_recommendations_batch(
    cases=cases_list,
    eligible_mediator_ids=[1, 2, 3],
    mediator_case_loads={1: 0, 2: 0, 3: 0},
    belief_state=belief_state,
    med_by_court_case_type=med_mapping,
    config=config,
    avg_case_rate=avg_case_rate,
    avg_p_val_by_crt_case_type=avg_p_val,
    court_stations=["MILIMANI", "KAKAMEGA"],
    case_types=["Family group"],
    generate_phantoms=True,
)
```

### Per-case Eligibility

By default a case may go to any mediator listed for its court station and case type in
`med_by_court_case_type`. Set `eligible_mediator_ids` on a case to replace that lookup for the
case — e.g. to exclude mediators who already declined it, or to apply rules the mapping can't
express (Kadhi-court religion, unavailability on the date). The list is still restricted to
the solver's `valid_mediators`.

```python
case = SimpleCase(id=42, case_type="Civil Cases", court_station="MILIMANI",
                  referral_date=today, p_value=0.55, eligible_mediator_ids=[3, 7, 12])
```

### Phantom Cases from Recent Arrivals

Phantom (future) cases can be drawn from recent real arrivals and scored with the fitted VA
model, so their mix of court stations, case types and other covariates — and their predicted
agreement probability — matches recent cases:

```python
import numpy as np
from smart_mediator_assignment import build_arrival_pool, generate_phantom_cases_from_pool

pool = build_arrival_pool(recent_cases, as_of=today, window_days=182)
phantoms, next_id = generate_phantom_cases_from_pool(
    current_day=today,
    time_horizon=config.time_horizon,
    pool=pool,
    va_model=va_result.model,      # from estimate_va
    rng=np.random.default_rng(seed),
    discount=0.1,                  # subtracted from each phantom's predicted p
)
results = get_recommendations_batch(..., phantom_cases=phantoms)
```

Each day draws a Poisson(`pool.daily_rate`) number of arrivals, then that many covariate vectors
uniformly from the pool. `va_result.model.predict(...)` scores a case at the phantom's arrival
month and the most recent quasiyear.

### VA Estimation (Batch)

Estimate mediator Value Added from historical case data using absorbing regression with shrinkage:

```python
from smart_mediator_assignment import (
    estimate_va,
    estimate_va_from_prepared,
    VAEstimationConfig,
)
from smart_mediator_assignment.core.case import CaseProtocol

# Cases must implement CaseProtocol
# (compatible with Django Case model from cadaster-kenya-mediation)
cases = get_historical_cases()  # Your data source

# Configure estimation
config = VAEstimationConfig(
    reference_date=datetime.now(),
    min_med_cases=2,
    days_since_appt_threshold=180,
)

# Run estimation
result = estimate_va(
    cases=cases,
    config=config,
    start_date="2016-04-06",
    end_date="2022-01-20",
)

# Access results
va_dict = result.get_va_dict()  # {mediator_id: va}
sigma = result.sigma            # Global sigma for belief initialization

# Individual mediator estimates
for med in result.mediator_vas:
    print(f"Mediator {med.mediator_id}: VA={med.va:.4f}, cases={med.n_cases}")

# Case-level predictions
for case in result.case_predictions:
    print(f"Case {case.case_id}: p_pred={case.p_pred:.4f}")

# The fitted model predicts p_pred for cases outside the fitted data
p = result.model.predict(case_type="Civil Cases", court_station="MILIMANI",
                         referral_mode="Referred by Court", court_type="Magistrate Court",
                         appt_month=3, quasiyear=0)
```

`estimate_va` always populates `result.prepared` with the cleaned, collapsed frame it fit
on. Reuse it with `estimate_va_from_prepared` to re-estimate on a **sub-window** of that frame
without re-cleaning the raw cases. Reuse can only select rows already present in `prepared`, so
it cannot widen the window — build the initial frame through your latest (reference) date, then
narrow:

```python
# result.prepared was built through the reference date above; re-estimate on a contained sub-window
refreshed = estimate_va_from_prepared(
    result.prepared,
    config=config,
    start_date="2016-04-06",
    end_date="2021-06-01",
)
```

The unified `CaseProtocol` supports both LP assignment and VA estimation. Required properties:
- `id`, `case_type`, `court_station`, `referral_date`, `p_value` (for assignment)
- `mediator_id`, `case_outcome_agreement`, `mediator_appointment_date`, `conclusion_date` (for VA estimation)
- `case_status`, `court_type`, `referral_mode` (for VA estimation)

## Testing

```bash
pytest tests/
```

## License

MIT
