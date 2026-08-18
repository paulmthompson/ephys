"""
Script to apply bandpass filtering and Whittaker-Shannon upsampling to binary electrophysiology data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Adjust imports according to the package structure.
try:
    from ephys.processing.filtering import design_intan_sos_bandpass, sos_bandpass_filter
    from ephys.processing.resampling import whittaker_shannon_interpolate
except ImportError:
    # If run as a standalone script without installing the package, add parent directory to path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ephys.processing.filtering import design_intan_sos_bandpass, sos_bandpass_filter
    from ephys.processing.resampling import whittaker_shannon_interpolate

UINT16_ZERO_OFFSET = 32768


def main():
    parser = argparse.ArgumentParser(
        description="Filter and upsample single-channel binary ephys data."
    )
    parser.add_argument("input_file", type=str, help="Path to the input binary file (.bin/.dat)")
    parser.add_argument("output_file", type=str, help="Path to save the output binary file")

    parser.add_argument("--fs", type=float, default=30000.0, help="Original sampling rate in Hz (default: 30000.0)")
    parser.add_argument("--up_factor", type=int, default=4, help="Upsampling factor (default: 4)")
    parser.add_argument(
        "--bit_depth",
        type=float,
        default=0.195,
        help="Bit depth in uV/bit after converting raw ADC counts to signed units (default: 0.195)",
    )
    parser.add_argument("--header_bytes", type=int, default=0, help="Number of bytes to skip in the header (default: 0)")
    parser.add_argument(
        "--uint16",
        action="store_true",
        help="Load input as uint16 with zero at 32768 (default: load as int16)",
    )

    # Filter options
    parser.add_argument("--lowcut", type=float, default=300.0, help="Bandpass lower cutoff in Hz (default: 300.0)")
    parser.add_argument("--highcut", type=float, default=5000.0, help="Bandpass upper cutoff in Hz (default: 5000.0)")
    parser.add_argument("--order", type=int, default=2, help="Bandpass filter order (default: 2)")
    parser.add_argument(
        "--filter",
        choices=["bessel", "butterworth"],
        default="bessel",
        help="Bandpass filter type (default: bessel)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    input_dtype = "uint16" if args.uint16 else "int16"
    print(f"Loading {input_dtype} data from {input_path}...")
    try:
        if args.uint16:
            data_raw = np.fromfile(input_path, dtype=np.uint16, offset=args.header_bytes)
        else:
            data_raw = np.fromfile(input_path, dtype=np.int16, offset=args.header_bytes)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Loaded {len(data_raw)} samples.")

    # Convert to float32 and scale to physical units (e.g., uV)
    print("Converting to float32 and scaling by bit depth...")
    if args.uint16:
        data_float = (data_raw.astype(np.float32) - UINT16_ZERO_OFFSET) * args.bit_depth
    else:
        data_float = data_raw.astype(np.float32) * args.bit_depth

    # Free up memory if possible
    del data_raw

    # Apply Bandpass Filter
    filter_label = args.filter.capitalize()
    print(
        f"Designing {args.order}th order {filter_label} bandpass filter "
        f"({args.lowcut}-{args.highcut} Hz)..."
    )
    sos = design_intan_sos_bandpass(
        lowcut_hz=args.lowcut,
        highcut_hz=args.highcut,
        sampling_rate_hz=args.fs,
        order=args.order,
        filter_type=args.filter,
    )

    print("Applying forward-backward bandpass filter...")
    data_filtered = sos_bandpass_filter(data_float, sos)

    # Free memory
    del data_float

    # Upsample
    print(f"Upsampling by a factor of {args.up_factor}x using Whittaker-Shannon interpolation...")
    data_upsampled = whittaker_shannon_interpolate(
        data_filtered,
        up_factor=args.up_factor,
        # We can leave window_half_len and beta to their defaults, which are appropriate for this use case
    )

    # Convert back to int16
    print("Re-scaling and converting back to int16...")
    # Divide by bit depth to get back to ADC units
    data_upsampled_int = (data_upsampled / args.bit_depth).astype(np.int16)

    # Save to disk
    print(f"Saving upsampled data to {output_path}...")
    try:
        data_upsampled_int.tofile(output_path)
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

    print(f"Done. Final number of samples: {len(data_upsampled_int)}")


if __name__ == "__main__":
    main()
