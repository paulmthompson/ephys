import pytest

from data_wrangling.ttls import find_index_of_ttl_event_from_another
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


def generate_alternating_transitions(length, step):
    """
    Generate a list of transitions alternating between 'low_to_high' and 'high_to_low'.

    Parameters
    ----------
    length: int
        The length of the array to generate.
    step: int
        The step size between transitions.

    Returns
    -------
    list of tuples
        Each tuple contains (index, transition_type) where transition_type is either 'low_to_high' or 'high_to_low'.
    """
    transitions = []
    transition_type = "low_to_high"
    for i in range(0, length, step):
        transitions.append((i, transition_type))
        transition_type = (
            "high_to_low" if transition_type == "low_to_high" else "low_to_high"
        )
    return transitions


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


def test_find_index_of_ttl_event_from_another():

    # Generate digital inputs with transitions for fake camera with alternating TTLs
    # alternate between 1 and 0 50 times every 2 samples
    length = 102
    transitions = generate_alternating_transitions(length, 2)
    camera_ttl = generate_digital_inputs_with_transitions(length, transitions)
    camera_ttl_onsets, ttl_offsets = get_ttl_timestamps_16bit(camera_ttl, 0)

    length = 100
    transitions = [
        (10, "low_to_high"),
        (20, "high_to_low"),
        (30, "low_to_high"),
        (40, "high_to_low"),
    ]
    laser_ttl = generate_digital_inputs_with_transitions(length, transitions)
    laser_ttl_onsets, laser_ttl_offsets = get_ttl_timestamps_16bit(laser_ttl, 0)

    laser_on_frames = find_index_of_ttl_event_from_another(
        camera_ttl_onsets,
        laser_ttl_onsets,
    )

    laser_off_frames = find_index_of_ttl_event_from_another(
        camera_ttl_onsets,
        laser_ttl_offsets,
    )

    expected_on_frames = np.array([2, 7])
    expected_off_frames = np.array([4, 9])

    assert np.array_equal(
        laser_on_frames, expected_on_frames
    ), f"Expected {expected_on_frames}, but got {laser_on_frames}"

    assert np.array_equal(
        laser_off_frames, expected_off_frames
    ), f"Expected {expected_off_frames}, but got {laser_off_frames}"

