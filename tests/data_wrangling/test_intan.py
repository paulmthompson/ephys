import pytest

from data_wrangling.intan import get_camera_ttl_array


def test_get_camera_ttl_array(pytestconfig):

    data_dir_path = pytestconfig.getoption("dirpath")
    intan_digital_filepath = f"{data_dir_path}/digitalin.dat"
    ttl_index = 0

    ttl_onsets, ttl_offsets = get_camera_ttl_array(
        intan_digital_filepath,
        ttl_index,
    )

    # Expected results
    num_expected_onsets = 1352

    assert ttl_onsets.shape[0] == num_expected_onsets

