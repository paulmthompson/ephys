import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

import sys

# Add project root to path so we can import 'ephys'
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from ephys.data_wrangling import intan
from ephys.processing.zca import apply_zca_whitening

def preprocess_intan_to_zca(
    input_filepath,
    output_filepath,
    channel_count=32,
    sampling_rate_hz=30000,
    lowcut=300.0,
    highcut=5000.0,
    dead_channels=None,
    epsilon=10.0
):
    """
    End-to-End preprocessing pipeline:
    1. Loads Intan binary data.
    2. Applies zero-phase SOS bandpass filtering.
    3. Excludes dead channels and applies robust ZCA whitening.
    4. Saves the results as Intan-compatible 16-bit integers to a new binary file.
    
    Args:
        input_filepath (str/Path): Path to raw amplifier.dat
        output_filepath (str/Path): Destination path for the whitened output .dat
        channel_count (int): Number of channels in the Intan recording
        sampling_rate_hz (float): Sampling rate in Hz
        lowcut (float): High-pass cutoff
        highcut (float): Low-pass cutoff
        dead_channels (list): List of channel indices to exclude from ZCA spatial mapping
        epsilon (float): Regularization parameter for the spatial whitening
    """
    if dead_channels is None:
        dead_channels = []
        
    print(f"Loading data from {input_filepath}...")
    voltage_uV = intan.load_voltage(str(input_filepath), channel_count)
    voltage_uV = np.swapaxes(voltage_uV, 0, 1)

    print(f"Applying SOS bandpass filter ({lowcut}-{highcut} Hz)...")
    nyq = 0.5 * sampling_rate_hz
    sos = butter(3, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    voltage_uV = sosfiltfilt(sos, voltage_uV, axis=1)

    print("Computing and applying robust ZCA (excluding dead channels)...")
    good_channels = [ch for ch in range(channel_count) if ch not in dead_channels]
    
    # Assign the result back to voltage_uV because advanced indexing creates a copy
    voltage_uV[good_channels, :] = apply_zca_whitening(
        voltage_uV[good_channels, :], 
        epsilon=epsilon, 
        rescale_amplitude=True,
        robust_cov=True
    )

    print("Converting to 16-bit Intan integers and saving...")
    INTAN_BIT_TO_uV = 0.195
    voltage_zca_int16 = np.round(voltage_uV / INTAN_BIT_TO_uV).astype(np.int16)
    voltage_zca_int16 = np.swapaxes(voltage_zca_int16, 0, 1)
    
    voltage_zca_int16.tofile(str(output_filepath))
    print(f"Preprocessing complete! Saved to {output_filepath}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="End-to-End preprocessing pipeline for Intan data")
    parser.add_argument("input_filepath", type=str, help="Path to raw amplifier.dat")
    parser.add_argument("output_filepath", type=str, help="Destination path for the whitened output .dat")
    parser.add_argument("--channel_count", type=int, default=32, help="Number of channels in the Intan recording")
    parser.add_argument("--sampling_rate_hz", type=float, default=30000.0, help="Sampling rate in Hz")
    parser.add_argument("--lowcut", type=float, default=300.0, help="High-pass cutoff")
    parser.add_argument("--highcut", type=float, default=5000.0, help="Low-pass cutoff")
    parser.add_argument("--dead_channels", type=int, nargs="*", default=None, help="List of channel indices to exclude")
    parser.add_argument("--epsilon", type=float, default=10.0, help="Regularization parameter")

    args = parser.parse_args()

    preprocess_intan_to_zca(
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        channel_count=args.channel_count,
        sampling_rate_hz=args.sampling_rate_hz,
        lowcut=args.lowcut,
        highcut=args.highcut,
        dead_channels=args.dead_channels,
        epsilon=args.epsilon
    )
