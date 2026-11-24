"""How the RCT treats a court station x case type cell missing from the randomized mapping."""
from smart_mediator_assignment import classify_unmapped_cell, mediator_treatment_shares


def test_share_is_the_fraction_of_a_mediators_cases_in_treatment_cells():
    shares = mediator_treatment_shares([
        (1, 'Treatment'), (1, 'Treatment'), (1, 'Control'),
        (2, 'Control'),
    ])
    assert shares == {1: (2 / 3, 3), 2: (0.0, 1)}


def test_cases_in_cells_without_an_arm_are_left_out():
    shares = mediator_treatment_shares([(1, 'Treatment'), (1, None), (3, None)])
    assert shares == {1: (1.0, 1)}   # mediator 3 has no counted case, so no share


def test_unknown_arm_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        mediator_treatment_shares([(1, 'Treated')])


def test_cell_is_treatment_when_its_mediators_lean_treatment():
    assert classify_unmapped_cell([1, 2], {1: 0.9, 2: 0.4}) == 'Treatment'   # average 0.65


def test_cell_is_control_when_its_mediators_lean_control():
    assert classify_unmapped_cell([1, 2], {1: 0.1, 2: 0.6}) == 'Control'     # average 0.35


def test_mediators_without_a_share_count_as_half():
    assert classify_unmapped_cell([1, 9], {1: 0.2}) == 'Control'             # (0.2 + 0.5) / 2
    assert classify_unmapped_cell([1, 9], {1: 0.8}) == 'Treatment'


def test_a_tie_defaults_to_treatment():
    assert classify_unmapped_cell([1, 2], {1: 0.3, 2: 0.7}) == 'Treatment'
    assert classify_unmapped_cell([7, 8], {}) == 'Treatment'                  # all new mediators


def test_a_cell_with_no_mediators_defaults_to_treatment():
    assert classify_unmapped_cell([], {1: 0.0}) == 'Treatment'
