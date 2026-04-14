import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

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
    good_voltage = voltage_uV[good_channels, :]
    
    voltage_zca_good = apply_zca_whitening(
        good_voltage, 
        epsilon=epsilon, 
        rescale_amplitude=True,
        robust_cov=True
    )
    
    # Map back to array with all channels 
    voltage_zca_full = np.copy(voltage_uV)
    voltage_zca_full[good_channels, :] = voltage_zca_good

    print("Converting to 16-bit Intan integers and saving...")
    INTAN_BIT_TO_uV = 0.195
    voltage_zca_int16 = np.round(voltage_zca_full / INTAN_BIT_TO_uV).astype(np.int16)
    voltage_zca_int16 = np.swapaxes(voltage_zca_int16, 0, 1)
    
    voltage_zca_int16.tofile(str(output_filepath))
    print(f"Preprocessing complete! Saved to {output_filepath}")
