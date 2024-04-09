import numpy as np


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


def get_low_to_high_transitions(ttl_boolean):

    # ERROR CHECKING
    # Values not equal to 0 and 1

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff > 0)[0] - 1

    return ttl_ticks


def get_high_to_low_transitions(ttl_boolean):

    # ERROR CHECKING
    # Values not equal to 0 and 1

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff < 0)[0] - 1

    return ttl_ticks
