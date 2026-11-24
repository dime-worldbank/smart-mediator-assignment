from .bayesian import compute_posterior, update_belief, update_belief_batch
from .phantom import (
    generate_phantom_cases,
    estimate_case_arrivals,
    ArrivalPool,
    ARRIVAL_WINDOW_DAYS,
    build_arrival_pool,
    generate_phantom_cases_from_pool,
)
from .strategies import (
    VAStrategy,
    MeanStrategy,
    SampleStrategy,
    GroundTruthStrategy,
    get_strategy,
)
from .va_estimation import (
    VAEstimationConfig,
    MediatorVAEstimate,
    CasePrediction,
    VAEstimationResult,
    VAModel,
    estimate_va,
    estimate_va_from_prepared,
)
from .case_types import (
    LEGACY_2_SIMPLIFIED_CTYPE,
    simplify_case_types,
)
from .belief_engine import (
    BeliefEngine,
    estimate_va_as_of,
    initial_mediator_sds,
)
from .unmapped_cells import classify_unmapped_cell, mediator_treatment_shares
from .duration_estimation import (
    estimate_lognormal_duration_params,
    clean_hazard_sample,
)

__all__ = [
    "compute_posterior",
    "update_belief",
    "update_belief_batch",
    "generate_phantom_cases",
    "ArrivalPool",
    "ARRIVAL_WINDOW_DAYS",
    "build_arrival_pool",
    "generate_phantom_cases_from_pool",
    "estimate_case_arrivals",
    "VAStrategy",
    "MeanStrategy",
    "SampleStrategy",
    "GroundTruthStrategy",
    "get_strategy",
    "VAEstimationConfig",
    "MediatorVAEstimate",
    "CasePrediction",
    "VAEstimationResult",
    "VAModel",
    "estimate_va",
    "estimate_va_from_prepared",
    # Case-type taxonomy (shared by VA + duration)
    "LEGACY_2_SIMPLIFIED_CTYPE",
    "simplify_case_types",
    # Belief engine (shared with the RCT simulation)
    "BeliefEngine",
    "estimate_va_as_of",
    "initial_mediator_sds",
    # Arms for cells missing from the RCT mapping
    "classify_unmapped_cell",
    "mediator_treatment_shares",
    # Duration estimation
    "estimate_lognormal_duration_params",
    "clean_hazard_sample",
]
