import numpy as np

from ephys.data_wrangling import intan

from ephys.processing.filtering import design_intan_sos_bandpass, sos_bandpass_filter

from ephys.processing.spatial_diagnostics import (
    compute_spatial_diagnostics,
    format_spatial_diagnostics_line,
    plan_spatial_subsample_indices,
)
from ephys.processing.zca import apply_zca_whitening

INTAN_BIT_TO_uV = 0.195


def apply_common_median_reference(voltage_uV, good_channels):
    """Subtract the across-channel median computed from good channels.

    Parameters
    ----------
    voltage_uV
        Voltage array with shape ``(n_channels, n_samples)``. Modified in place.
    good_channels
        Channel indices used to estimate the common median reference.

    Returns
    -------
    numpy.ndarray
        The same array as ``voltage_uV``.
    """
    reference = np.median(voltage_uV[good_channels, :], axis=0, keepdims=True)
    voltage_uV[good_channels, :] -= reference
    return voltage_uV


def preprocess_intan(
    input_filepath,
    output_filepath,
    channel_count=32,
    sampling_rate_hz=30000,
    lowcut=300.0,
    highcut=5000.0,
    filter_type="bessel",
    order=2,
    dead_channels=None,
    spatial_reference="zca",
    epsilon=10.0,
):
    """

    End-to-End preprocessing pipeline:

    1. Loads Intan binary data.

    2. Applies zero-phase SOS bandpass filtering.

    3. Applies spatial reference on good channels (ZCA, CMR, or CMR then ZCA).

    4. Saves the results as Intan-compatible 16-bit integers to a new binary file.

    Args:

        input_filepath (str/Path): Path to raw amplifier.dat

        output_filepath (str/Path): Destination path for the processed output .dat

        channel_count (int): Number of channels in the Intan recording

        sampling_rate_hz (float): Sampling rate in Hz

        lowcut (float): High-pass cutoff

        highcut (float): Low-pass cutoff

        filter_type (str): Bandpass design to use ("bessel" or "butterworth")

        order (int): Bandpass filter order

        dead_channels (list): List of channel indices excluded from spatial reference

        spatial_reference (str): One of ``"zca"``, ``"cmr"``, or ``"cmr_zca"``

        epsilon (float): Regularization parameter for ZCA whitening

    """

    if dead_channels is None:
        dead_channels = []

    if spatial_reference not in ("zca", "cmr", "cmr_zca"):
        msg = (
            f"spatial_reference must be 'zca', 'cmr', or 'cmr_zca'; "
            f"got {spatial_reference!r}"
        )
        raise ValueError(msg)

    print(f"Loading data from {input_filepath}...")
    voltage_uV = intan.load_voltage(str(input_filepath), channel_count)
    voltage_uV = np.swapaxes(voltage_uV, 0, 1)

    print(
        f"Applying {order}th order {filter_type.capitalize()} SOS bandpass filter "
        f"({lowcut}-{highcut} Hz)..."
    )

    sos = design_intan_sos_bandpass(
        lowcut_hz=lowcut,
        highcut_hz=highcut,
        sampling_rate_hz=sampling_rate_hz,
        order=order,
        filter_type=filter_type,
    )

    voltage_uV = sos_bandpass_filter(voltage_uV, sos, axis=1)

    good_channels = [ch for ch in range(channel_count) if ch not in dead_channels]
    diagnostic_indices = plan_spatial_subsample_indices(voltage_uV.shape[1])

    def log_spatial_diagnostics(label: str) -> None:
        diagnostics = compute_spatial_diagnostics(
            voltage_uV,
            good_channels,
            diagnostic_indices,
        )
        print(format_spatial_diagnostics_line(label, diagnostics))

    print("Spatial diagnostics (subsampled; good channels only):")
    log_spatial_diagnostics("after bandpass")

    if spatial_reference in ("cmr", "cmr_zca"):
        print(
            "Applying common median reference (CMR) on good channels "
            f"(excluding dead channels: {dead_channels or 'none'})..."
        )
        apply_common_median_reference(voltage_uV, good_channels)
        log_spatial_diagnostics("after CMR")

    if spatial_reference in ("zca", "cmr_zca"):
        step = "robust ZCA after CMR" if spatial_reference == "cmr_zca" else "robust ZCA"
        print(f"Computing and applying {step} (excluding dead channels)...")
        voltage_uV[good_channels, :] = apply_zca_whitening(
            voltage_uV[good_channels, :],
            epsilon=epsilon,
            rescale_amplitude=True,
            robust_cov=True,
            sampling_rate_hz=sampling_rate_hz,
        )
        log_spatial_diagnostics("after ZCA")

    print("Converting to 16-bit Intan integers and saving...")

    voltage_int16 = np.round(voltage_uV / INTAN_BIT_TO_uV).astype(np.int16)
    voltage_int16 = np.swapaxes(voltage_int16, 0, 1)
    voltage_int16.tofile(str(output_filepath))

    print(f"Preprocessing complete! Saved to {output_filepath}")


def preprocess_intan_to_zca(*args, **kwargs):
    """Backward-compatible alias for :func:`preprocess_intan`."""
    return preprocess_intan(*args, **kwargs)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-End preprocessing pipeline for Intan data"
    )
    parser.add_argument("input_filepath", type=str, help="Path to raw amplifier.dat")
    parser.add_argument(
        "output_filepath",
        type=str,
        help="Destination path for the processed output .dat",
    )
    parser.add_argument(
        "--channel_count",
        type=int,
        default=32,
        help="Number of channels in the Intan recording",
    )
    parser.add_argument(
        "--sampling_rate_hz", type=float, default=30000.0, help="Sampling rate in Hz"
    )
    parser.add_argument("--lowcut", type=float, default=300.0, help="High-pass cutoff")
    parser.add_argument("--highcut", type=float, default=5000.0, help="Low-pass cutoff")
    parser.add_argument(
        "--filter",
        choices=["bessel", "butterworth"],
        default="bessel",
        help="Bandpass filter type (default: bessel)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Bandpass filter order (default: 2)",
    )
    parser.add_argument(
        "--dead_channels",
        type=int,
        nargs="*",
        default=None,
        help="List of channel indices to exclude from spatial reference",
    )
    parser.add_argument(
        "--spatial-reference",
        choices=["zca", "cmr", "cmr_zca"],
        default="zca",
        help=(
            "Spatial reference mode: zca (default), cmr (common median reference only), "
            "or cmr_zca (CMR followed by ZCA)"
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=10.0,
        help="ZCA regularization parameter (used with zca and cmr_zca)",
    )

    args = parser.parse_args()

    preprocess_intan(
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        channel_count=args.channel_count,
        sampling_rate_hz=args.sampling_rate_hz,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_type=args.filter,
        order=args.order,
        dead_channels=args.dead_channels,
        spatial_reference=args.spatial_reference,
        epsilon=args.epsilon,
    )
