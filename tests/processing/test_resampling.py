import numpy as np

from ephys.processing.resampling import whittaker_shannon_interpolate


def test_whittaker_shannon_interpolate_upsample():
    """Test that upsampling interpolates correctly without distortion."""
    # Create a synthetic bandlimited signal (e.g., sum of sine waves)
    fs = 30000.0
    t = np.arange(0, 0.1, 1 / fs)  # 100 ms of data

    # 300 Hz and 1000 Hz components
    f1, f2 = 300.0, 1000.0
    signal = 10 * np.sin(2 * np.pi * f1 * t) + 5 * np.cos(2 * np.pi * f2 * t)

    up_factor = 4
    upsampled_signal = whittaker_shannon_interpolate(signal, up_factor=up_factor)

    # Check shape
    assert upsampled_signal.shape[0] == len(signal) * up_factor

    # Generate the ideal high-resolution signal
    fs_high = fs * up_factor
    t_high = np.arange(0, 0.1, 1 / fs_high)
    ideal_signal = 10 * np.sin(2 * np.pi * f1 * t_high) + 5 * np.cos(2 * np.pi * f2 * t_high)

    # The interpolation introduces some edge effects, so we compare the middle portion
    margin = int(0.01 * fs_high)  # 10 ms margin

    error = np.abs(upsampled_signal[margin:-margin] - ideal_signal[margin:-margin])

    # The error should be very small for a clean bandlimited signal well below Nyquist
    # Using polyphase FIR, we expect the difference to be < 0.1% of signal amplitude
    max_error = np.max(error)
    assert max_error < 0.1, f"Max error {max_error} is too high"

def test_whittaker_shannon_interpolate_multidimensional():
    """Test that upsampling works on multidimensional arrays."""
    # Create a 2D array: 5 channels, 1000 samples
    signal = np.random.randn(1000, 5)

    up_factor = 2
    upsampled = whittaker_shannon_interpolate(signal, up_factor=up_factor, axis=0)

    assert upsampled.shape == (2000, 5)
