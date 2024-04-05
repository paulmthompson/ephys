import numpy as np

from .binary_data import get_digital
from .ttls import find_high_ttls_at_single_channel


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

    ttl_boolean = find_high_ttls_at_single_channel(digital_inputs, ttl_index)

    ttl_boolean_diff = np.ediff1d(ttl_boolean)
    ttl_ticks = np.where(ttl_boolean_diff > 0)[0] - 1
    return ttl_ticks


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
