import pytest

from data_wrangling.ttls import get_high_to_low_transition_timestamps
from data_wrangling.ttls import get_low_to_high_transition_timestamps
from data_wrangling.ttls import get_ttl_timestamps_16bit

import numpy as np


def generate_digital_inputs_with_transitions(length, transitions):
    """
    Generate an array of uint16 numbers with specified transitions.

    Parameters
    ----------
    length: int
        Length of the array to generate.
    transitions: list of tuples
        Each tuple contains (index, transition_type) where transition_type is either 'low_to_high' or 'high_to_low'.

    Returns
    -------
    np.ndarray[np.uint16]
        Array of uint16 numbers with specified transitions.
    """
    digital_inputs = np.zeros(length, dtype=np.uint16)
    for index, transition_type in transitions:
        if transition_type == "low_to_high":
            digital_inputs[index:] = 1
        elif transition_type == "high_to_low":
            digital_inputs[index:] = 0
    return digital_inputs


def test_get_low_to_high_transition_timestamps():
    # Generate digital inputs with transitions
    length = 100
    transitions = [
        (10, "low_to_high"),
        (20, "high_to_low"),
        (30, "low_to_high"),
        (40, "high_to_low"),
    ]
    digital_inputs = generate_digital_inputs_with_transitions(length, transitions)

    # Test the function
    ttl_onsets = get_low_to_high_transition_timestamps(digital_inputs)

    # Expected results
    expected_onsets = np.array([10, 30])

    assert np.array_equal(
        ttl_onsets, expected_onsets
    ), f"Expected {expected_onsets}, but got {ttl_onsets}"


def test_get_high_to_low_transition_timestamps():
    # Generate digital inputs with transitions
    length = 100
    transitions = [
        (10, "low_to_high"),
        (20, "high_to_low"),
        (30, "low_to_high"),
        (40, "high_to_low"),
    ]
    digital_inputs = generate_digital_inputs_with_transitions(length, transitions)

    # Test the function
    ttl_offsets = get_high_to_low_transition_timestamps(digital_inputs)

    # Expected results
    expected_offsets = np.array([20, 40])

    assert np.array_equal(
        ttl_offsets, expected_offsets
    ), f"Expected {expected_offsets}, but got {ttl_offsets}"


def test_get_ttl_timestamps_16bit():
    # Generate digital inputs with transitions
    length = 100
    transitions = [
        (10, "low_to_high"),
        (20, "high_to_low"),
        (30, "low_to_high"),
        (40, "high_to_low"),
    ]
    digital_inputs = generate_digital_inputs_with_transitions(length, transitions)

    # Test the function
    ttl_index = 0
    ttl_onsets, ttl_offsets = get_ttl_timestamps_16bit(digital_inputs, ttl_index)

    # Expected results
    expected_onsets = np.array([10, 30])
    expected_offsets = np.array([20, 40])

    assert np.array_equal(
        ttl_onsets, expected_onsets
    ), f"Expected {expected_onsets}, but got {ttl_onsets}"
    assert np.array_equal(
        ttl_offsets, expected_offsets
    ), f"Expected {expected_offsets}, but got {ttl_offsets}"
