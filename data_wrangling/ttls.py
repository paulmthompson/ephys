import numpy as np


def find_index_of_ttl_event_from_another(ttl_events_1, ttl_events_2,):
    """
    This function will find the closest index of each ttl event in ttl_events_1
    for each event in ttl_events_2.

    This function is useful for example if you recorded camera frames with ttl_events_1,
    and some experimental event with ttl_events_2, and you want to know which camera frame
    corresponds to each experimental event.

    Parameters
    ----------
    ttl_events_1
    ttl_events_2

    Returns
    -------

    """

    output_event_indices = []
    for event in ttl_events_2:
        closest_index = np.argmin(np.abs(ttl_events_1 - event))
        output_event_indices.append(closest_index)

    return output_event_indices



def get_ttl_timestamps_16bit(
    digital_inputs,
    ttl_index,
    isTransitionLowToHigh=None,
):
    """

    Parameters
    ----------
    digital_inputs: np.ndarray[np.uint16]
        Single 16-bit number representing 16-bit digital inputs for each sample
    ttl_index: int
        zero based index (0-15). This is the index of the TTL
        channel that we are interested in
    isTransitionLowToHigh: bool or None
        If None, the function will attempt to determine
        the transition type from the data
    Returns
    -------
    np.ndarray[int]:
        Timestamps for the start of each event, adjusted so
        that each has a matching off timestamp
    np.ndarray[int]
        Timestamps for the end of each event, adjusted so
        that each has a matching on timestamp
    """

    ttl_boolean = find_ttls_on_single_channel_16bit(digital_inputs, ttl_index)

    if isTransitionLowToHigh is None:
        isTransitionLowToHigh = is_ttl_transition_low_to_high(ttl_boolean)

    lowToHighTimestamps = get_low_to_high_transition_timestamps(ttl_boolean)

    highToLowTimestamps = get_high_to_low_transition_timestamps(ttl_boolean)

    if isTransitionLowToHigh:
        return match_ttl_timestamps(lowToHighTimestamps, highToLowTimestamps)
    else:
        return match_ttl_timestamps(highToLowTimestamps, lowToHighTimestamps)


def match_ttl_timestamps(
    transition_1_timestamps,
    transition_2_timestamps,
):
    """

    Parameters
    ----------
    transition_1_timestamps: np.ndarray[int]
        These are the timestamps for the start of
        each event
    transition_2_timestamps: np.ndarray[int]
        These are the timestamps for the end of
        each event

    Returns
    -------
    np.ndarray[int]:
        Timestamps for the start of each event, adjusted so
        that each has a matching off timestamp
    np.ndarray[int]
        Timestamps for the end of each event, adjusted so
        that each has a matching on timestamp
    """

    num_transition_1 = len(transition_1_timestamps)
    num_transition_2 = len(transition_2_timestamps)

    if num_transition_1 != num_transition_2:
        print("The number of transition timestamps are not equal")
        print("Transition 1 timestamps: " + str(num_transition_1))
        print("Transition 2 timestamps: " + str(num_transition_2))

    t1_offset = 0
    if transition_1_timestamps[0] > transition_2_timestamps[0]:
        print(
            "The first event appears to be the end of a transition. \
              This means that the first event was mostly likely cut off"
        )
        t1_offset = 1

    t2_offset = None
    if transition_2_timestamps[-1] < transition_2_timestamps[-1]:
        print(
            "The last event appears to have no matching end transition "
            "This means the last event was most likely cut off"
        )
        t2_offset = -1

    transition_durations = (
        transition_2_timestamps[:t2_offset] - transition_1_timestamps[t1_offset:]
    )

    print(f"The TTL duration appears to be {np.median(transition_durations)} samples")

    return transition_1_timestamps[t1_offset:], transition_2_timestamps[:t2_offset]


def find_ttls_on_single_channel_16bit(digital_inputs, ttl_index):
    """

    TTL signals from multiple channels are generally packed
    together in a single binary number, where each 0 or 1
    corresponds to a single channel being high or low.

    This function is used when the digital inputs are packed
    together in a 16-bit number; consequently there can be
    16 possible TTL channels for each sample.


    Parameters
    ----------
    digital_inputs: np.ndarray[np.uint16]
        Single 16-bit number representing 16-bit digital inputs for each sample
    ttl_index: int
        zero based index (0-15)

    Returns
    -------

    """

    if (ttl_index < 0) or (ttl_index > 15):
        raise ValueError("TTL index must be between 0 and 15")

    binary_ttl_mask = 2**ttl_index
    ttl_boolean = ((digital_inputs & binary_ttl_mask) > 0).astype(int)

    return ttl_boolean


def is_ttl_transition_low_to_high(ttl_boolean):
    """

    Depending on the configuration of the attached electronics,
    a TTL input may rest in the high state, and temporarily
    transition to the low state during an event, or vice versa.

    This method assumes that the rest state is the one where
    the input most often resides (e.g. if the input is usually high,
    then high to low transitions signal events).

    Parameters
    ----------
    ttl_boolean: np.ndarray[int]
        Array of samples from single ttl channel.
        0 represents low and 1 represents high

    Returns
    -------
    bool:
        True indicates that this TTL input
        is most likely configured for inputs that are
        low to high
    """

    total_high_ttl = np.count_nonzero(ttl_boolean)
    total_low_ttl = len(ttl_boolean) - total_high_ttl

    print("Total High TTL Values: ", total_high_ttl)
    print("Total Low TTL Values: ", total_low_ttl)
    if total_high_ttl > total_low_ttl:
        print("Events are most likely to be on high to low transition")
        return False
    else:
        print("Events are likely to be on low to high transition")
        return True


def get_low_to_high_transition_timestamps(ttl_boolean):
    """

    Parameters
    ----------
    ttl_boolean: np.ndarray[int]
        Array of samples from single ttl channel.
        0 represents low and 1 represents high

    Returns
    -------
    np.ndarray[int]:
        Time stamps (in index values) of transitions
        between low and high values
    """

    if not np.all(np.isin(ttl_boolean, [0, 1])):
        raise ValueError("ttl_boolean array must contain only 0 and 1 values")

    # Make sure ttl boolean is signed
    ttl_boolean = np.array(ttl_boolean, dtype=int)

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff > 0)[0] + 1

    return ttl_ticks


def get_high_to_low_transition_timestamps(ttl_boolean):
    """

    Parameters
    ----------
    ttl_boolean: np.ndarray[int]
        Array of samples from single ttl channel.
        0 represents low and 1 represents high

    Returns
    -------
    np.ndarray[int]:
        Time stamps (in index values) of transitions
        between high and low values
    """

    if not np.all(np.isin(ttl_boolean, [0, 1])):
        raise ValueError("ttl_boolean array must contain only 0 and 1 values")

    # Make sure ttl boolean is signed
    ttl_boolean = np.array(ttl_boolean, dtype=int)

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff < 0)[0] + 1

    return ttl_ticks
