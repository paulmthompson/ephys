import numpy as np
import argparse
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
import sys

# Add project root to path so we can import 'ephys'
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from ephys.data_wrangling import intan
from ephys.processing.zca import apply_zca_whitening
from ephys.probes import get_probe

import spikeinterface.extractors as se
from spikeinterface.preprocessing import correct_motion
from spikeinterface.sortingcomponents.motion import interpolate_motion


def preprocess_motion_zca(
    input_filepath,
    output_filepath,
    channel_count=32,
    sampling_rate_hz=30000.0,
    lowcut=300.0,
    highcut=5000.0,
    epsilon=0.1,
    motion_preset="dredge",
    border_mode="force_zeros",
    spatial_interpolation_method="kriging",
    probe_type="poly2",
    dead_channels=None,
):
    """
    Advanced preprocessing pipeline incorporating motion correction and ZCA whitening.
    Branch 1: Estimates motion using a CMR filter.
    Branch 2: Applies ZCA spatial whitening to raw bandpassed data, then interpolates motion.
    """
    if dead_channels is None:
        dead_channels = []

    good_channels = [ch for ch in range(channel_count) if ch not in dead_channels]

    print(f"Loading data from {input_filepath}...")
    voltage_uV = intan.load_voltage(str(input_filepath), channel_count)
    voltage_uV = np.swapaxes(voltage_uV, 0, 1)

    print(f"Applying SOS bandpass filter ({lowcut}-{highcut} Hz)...")
    nyq = 0.5 * sampling_rate_hz
    sos = butter(3, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    voltage_uV_filtered = sosfiltfilt(sos, voltage_uV, axis=1)

    print("Applying CMR filter for motion estimation branch (excluding dead channels)...")
    cmr_median = np.median(voltage_uV_filtered[good_channels, :], axis=0)
    voltage_uV_cmr = voltage_uV_filtered - cmr_median

    print(f"Setting up SpikeInterface recording for motion estimation (Probe: {probe_type})...")
    recording_cmr = se.NumpyRecording(
        traces_list=voltage_uV_cmr.transpose(), sampling_frequency=sampling_rate_hz
    )
    probe = get_probe(probe_type)
    recording_cmr = recording_cmr.set_probe(probe)

    print(f"Estimating motion vectors using preset '{motion_preset}'...")
    _, motion, _ = correct_motion(
        recording=recording_cmr, preset=motion_preset, output_motion=True, output_motion_info=True
    )

    print("Applying robust ZCA whitening to bandpass filtered data (excluding dead channels)...")
    voltage_uV_zca = voltage_uV_filtered.copy()
    voltage_uV_zca[good_channels, :] = apply_zca_whitening(
        voltage_uV_filtered[good_channels, :],
        epsilon=epsilon,
        rescale_amplitude=True,
        robust_cov=True,
    )

    print("Setting up SpikeInterface recording for interpolation...")
    recording_zca = se.NumpyRecording(
        traces_list=voltage_uV_zca.transpose(), sampling_frequency=sampling_rate_hz
    )
    recording_zca = recording_zca.set_probe(probe)

    print(f"Applying motion correction (interpolation) using {spatial_interpolation_method}...")
    recording_motion_corrected = interpolate_motion(
        recording=recording_zca,
        motion=motion,
        border_mode=border_mode,
        spatial_interpolation_method=spatial_interpolation_method,
    )

    voltage_uV_final = recording_motion_corrected.get_traces().transpose()

    print("Converting to 16-bit Intan integers and saving...")
    INTAN_BIT_TO_uV = 0.195
    v_new_int16 = np.round(voltage_uV_final / INTAN_BIT_TO_uV).astype(np.int16)
    v_new_int16 = np.swapaxes(v_new_int16, 0, 1)

    v_new_int16.tofile(str(output_filepath))
    print(f"Preprocessing complete! Saved to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZCA Whitening and Motion Correction Pipeline")
    parser.add_argument("input_filepath", type=str, help="Path to raw amplifier.dat")
    parser.add_argument(
        "output_filepath",
        type=str,
        help="Destination path for the whitened and motion-corrected output .dat",
    )
    parser.add_argument("--channel_count", type=int, default=32, help="Number of channels")
    parser.add_argument(
        "--sampling_rate_hz", type=float, default=30000.0, help="Sampling rate in Hz"
    )
    parser.add_argument("--lowcut", type=float, default=300.0, help="High-pass cutoff")
    parser.add_argument("--highcut", type=float, default=5000.0, help="Low-pass cutoff")
    parser.add_argument("--epsilon", type=float, default=0.1, help="ZCA Regularization parameter")
    parser.add_argument(
        "--motion_preset", type=str, default="dredge", help="Motion estimation preset"
    )
    parser.add_argument(
        "--border_mode", type=str, default="force_zeros", help="Border mode for interpolation"
    )
    parser.add_argument(
        "--spatial_interpolation_method",
        type=str,
        default="kriging",
        help="Spatial interpolation method",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="poly2",
        choices=["poly2", "poly3"],
        help="Probe geometry layout",
    )
    parser.add_argument(
        "--dead_channels",
        type=int,
        nargs="*",
        default=None,
        help="List of channel indices to exclude",
    )

    args = parser.parse_args()

    preprocess_motion_zca(
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        channel_count=args.channel_count,
        sampling_rate_hz=args.sampling_rate_hz,
        lowcut=args.lowcut,
        highcut=args.highcut,
        epsilon=args.epsilon,
        motion_preset=args.motion_preset,
        border_mode=args.border_mode,
        spatial_interpolation_method=args.spatial_interpolation_method,
        probe_type=args.probe_type,
        dead_channels=args.dead_channels,
    )
