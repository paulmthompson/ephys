import numpy as np


def get_ttl_timestamps(digital_inputs, ttl_index):

    ttl_boolean = find_high_ttls_at_single_channel(digital_inputs, ttl_index)

    isTransitionLowToHigh = is_ttl_transition_low_to_high(ttl_boolean)

    lowToHighTimestamps = get_low_to_high_transition_timestamps(ttl_boolean)

    highToLowTimestamps = get_low_to_high_transition_timestamps(ttl_boolean)

    if isTransitionLowToHigh:
        return match_ttl_timestamps(lowToHighTimestamps, highToLowTimestamps)
    else:
        return match_ttl_timestamps(highToLowTimestamps, lowToHighTimestamps)


def match_ttl_timestamps(
    transition_1_timestamps,
    transition_2_timestamps,
):

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

    t2_offset = 0
    if transition_2_timestamps[-1] < transition_2_timestamps[-1]:
        print(
            "The last event appears to have no matching end transition "
            "This means the last event was most likely cut off"
        )
        t2_offset = -1

    transition_durations = transition_2_timestamps[:t2_offset] - transition_1_timestamps[t1_offset:]

    print(f"The TTL duration appears to be {np.median(transition_durations)} samples")

    return transition_1_timestamps[t1_offset:], transition_2_timestamps[:t2_offset]



def find_high_ttls_at_single_channel(digital_inputs, ttl_index):
    """



    Parameters
    ----------
    digital_inputs: np.ndarray[np.uint16]
        Single 16-bit number representing 16-bit digital inputs for each sample
    ttl_index: int
        zero based index (0-15)

    Returns
    -------

    """

    # ERROR CHECKING
    # Make sure ttl_index is not larger than number of bits in digital input
    # type (does this matter if python doesn't care so much about types?)

    binary_ttl_mask = 2**ttl_index
    ttl_boolean = ((digital_inputs & binary_ttl_mask) > 0).astype(int)

    return ttl_boolean


def is_ttl_transition_low_to_high(ttl_boolean):
    """

    Parameters
    ----------
    ttl_boolean

    Returns
    -------

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

    # ERROR CHECKING
    # Values not equal to 0 and 1

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff > 0)[0] - 1

    return ttl_ticks


def get_high_to_low_transition_timestamps(ttl_boolean):

    # ERROR CHECKING
    # Values not equal to 0 and 1

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff < 0)[0] - 1

    return ttl_ticks
