
import pickle

"""
These codes are for loading spikes output from Neuroviz manual sorting program.

https://gitlab.oit.duke.edu/herzfeldd/NeuroViz.jl
"""

def get_spikes(file_path, sampling_frequency=30000.):
    """

    Get spike times (in ticks) from neuroviz file

    Parameters
    ----------
    file_path: string
        Path to the neuroviz pickle file
    sampling_frequency: float
        unused

    Returns
    -------
    List
        Each element is a np.ndarray[int] corresponding
        to spike times from a different unit.
    """

    units_ = open_neuroviz(file_path)

    spike_times_for_each_neuron = []

    for i in range(len(units_)):
        spike_times_for_each_neuron.append(
            units_[i]['spike_indices__'])

    return spike_times_for_each_neuron


def open_neuroviz(filepath):
    """

    Open neuroviz pickle file. Each entry in list is a different neuron

    Parameters
    ----------
    filepath: string
        path to pickle file

    Returns
    -------
    List
        Each element corresponds to a unique neuron
    """

    units_file = open(filepath, 'rb')
    units = pickle.load(units_file)

    return units
