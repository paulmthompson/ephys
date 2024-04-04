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
    header_offset_in_bytes: int
        Binary data file may have header information
    single_sample_size_in_bytes:int

    Returns
    -------
    np.ndarray

    """

    match single_sample_size_in_bytes:
        case 1:
            d_type = np.uint8
        case 2:
            d_type = np.uint16
        case _:
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
