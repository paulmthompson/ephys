import numpy as np
import copy


def get_spikes_at_events(
    spike_times_ticks,
    event_ticks,
    win_ticks,
    sampling_rate_hz=30000,
):
    """
    Find the spikes that occur within a window around events

    Parameters
    ----------
    spike_times_ticks: np.ndarray
    event_ticks: np.ndarray
    win_ticks: int
    sampling_rate_hz: int

    Returns
    -------
    list:

    """

    spikes_in_range_s = []

    for event_tick in event_ticks:
        event_lower_bound_tick = event_tick - win_ticks
        event_upper_bound_tick = event_tick + win_ticks

        spikes_in_range_ticks = np.take(
            spike_times_ticks,
            np.where(
                (event_lower_bound_tick < spike_times_ticks)
                & (spike_times_ticks < event_upper_bound_tick)
            ),
        )[0]

        spikes_in_range_s.append(
            spikes_in_range_ticks / sampling_rate_hz - event_tick / sampling_rate_hz
        )

    return spikes_in_range_s


def sort_by_spike_times(spike_times):
    """
    Given a list of spike times for each trial, return the indices of the trials
    sorted by the latency to the first spike time

    Parameters
    ----------
    spike_times: list
        spike times for each trial

    Returns
    -------
    list:
        indices of trials sorted by latency to first spike time

    """

    sorted_spike_times = []

    for i in range(0, len(spike_times)):
        # Get the first spike time that is greater than zero
        first_spike_time = np.where(spike_times[i] > 0)[0]
        if len(first_spike_time) > 0:
            sorted_spike_times.append(spike_times[i][first_spike_time[0]])
        else:
            sorted_spike_times.append(0.0)

    sorted_order = np.argsort(sorted_spike_times)
    return sorted_order


def remove_spike_times_after_event(spiketimes_s, event_s):
    """
    Remove spike times that occur after an event

    Parameters
    ----------
    spiketimes_s: list
        spike times per trial in seconds
    event_s: list
        event times per trial in seconds

    Returns
    -------
    list:
        copy of spiketimes_s with spike times after event removed

    """

    # Check that the number of trials is the same
    if len(spiketimes_s) != len(event_s):
        raise ValueError(
            "The number of trials in spiketimes_s and event_s must be the same."
        )

    spiketimes_s_copy = copy.deepcopy(spiketimes_s)

    for i in range(0, len(spiketimes_s_copy)):
        spiketimes_s_copy[i] = spiketimes_s_copy[i][spiketimes_s_copy[i] < event_s[i]]

    return spiketimes_s_copy
