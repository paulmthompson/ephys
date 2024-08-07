import numpy as np


def get_digital(
    filepath,
    header_offset_in_bytes=0,
    single_sample_size_in_bytes=2,
):
    """

    Parameters
    ----------
    filepath: string
        Path to the binary data file
    header_offset_in_bytes: int
        Binary data file may have header information
    single_sample_size_in_bytes: int
        Size of a single sample in bytes
    Returns
    -------
    np.ndarray
        Array of digital inputs
    """

    # match statement introduced in 3.10
    if single_sample_size_in_bytes == 1:
        d_type = np.uint8
    elif single_sample_size_in_bytes == 2:
        d_type = np.uint16
    else:
        raise ValueError(
            "Unsupported digital input byte size. Are \
                you sure that you have more than 16 channels?"
        )

    digital_inputs = np.fromfile(
        filepath,
        dtype=d_type,
        offset=header_offset_in_bytes,
    )

    print(f"Number of samples: {len(digital_inputs)}")

    return digital_inputs

