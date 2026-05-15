import numpy as np

def apply_zca_whitening(voltage_matrix, epsilon=10.0, rescale_amplitude=True, robust_cov=True):
    """
    Applies ZCA Spatial Whitening to a (channels x samples) voltage matrix.
    Uses robust MAD (Median Absolute Deviation) to evaluate variance scaling and prevent
    artifact blowout. Can optionally calculate the covariance matrix purely from
    artifact-free baseline data using `robust_cov=True`.
    """
    # 1. To avoid redundant extreme sorting, compute median once and use it everywhere.
    # Note: For bandpassed ephys data, median is effectively 0, but this ensures perfect centering.
    median_vars = np.median(voltage_matrix, axis=1, keepdims=True)
    voltage_matrix -= median_vars

    # 2. Compute a unified, cached Robust Standard Deviation (MAD)
    # Because centered_volt is cleanly centered, we completely skip the nested inner np.median calculation
    robust_std = np.median(np.abs(voltage_matrix), axis=1, keepdims=True) * 1.4826

    # 3. Compute covariance matrix across channels
    if robust_cov:
        # Find time-samples where ALL channels are below 4x std to drop movement artifacts
        is_clean_sample = np.all(np.abs(voltage_matrix) < (4 * robust_std), axis=0)
        clean_centered_volt = voltage_matrix[:, is_clean_sample]
        cov = np.cov(clean_centered_volt)
    else:
        cov = np.cov(voltage_matrix)

    # 4. Eigen decomposition
    U, S, _ = np.linalg.svd(cov)

    # 5. Compute ZCA matrix with epsilon regularization
    ZCA_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T

    # 6. Apply ZCA
    voltage_matrix[:] = ZCA_matrix @ voltage_matrix

    # 7. Rescale back to approximate original amplitude using our unified robust MAD
    if rescale_amplitude:
        fixed_scalar = np.mean(robust_std)
        voltage_matrix *= fixed_scalar

    return voltage_matrix
