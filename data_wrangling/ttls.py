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
    # type (does this matter if python doens't care so much about types?)

    binary_ttl_mask = 2**ttl_index
    ttl_boolean = ((digital_inputs & binary_ttl_mask) > 0).astype(int)

    return ttl_boolean
