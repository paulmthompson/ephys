import numpy as np

from .binary_data import get_digital
from .ttls import get_ttl_timestamps_16bit


def get_camera_ttl_array(intan_digital_filepath, ttl_index=1):
    """
    Intan stores 16 digital inputs in one 16-bit number per timestamp sampled at
    the system sampling rate (usually 30,000 Hz). Here we first find the binary
    High/low value of a single TTL based on index, and then specifically find
    the times (in ticks) where that TTL transitions from low to high

    Parameters
    ----------
    intan_digital_filepath: string
    ttl_index: int

    Returns
    -------
    np.ndarray[int]:
        Time (in ticks) where ttl at index transitions from low to high
    """
    digital_inputs = get_digital(
        intan_digital_filepath,
        0,
        2,
    )

    ttl_onsets, ttl_offsets = get_ttl_timestamps_16bit(digital_inputs, ttl_index)

    return ttl_onsets


def load_voltage(voltage_filepath, channel_count):
    """

    Parameters
    ----------
    voltage_filepath: string
    channel_count: int

    Returns
    -------

    """
    voltage = np.fromfile(voltage_filepath, dtype=np.int16)
    voltage = voltage.reshape(channel_count, int(voltage.shape[0] / channel_count))
    voltage_uV = voltage * 0.195
    return voltage_uV
