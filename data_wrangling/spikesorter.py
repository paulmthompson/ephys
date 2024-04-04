import csv
import numpy as np
from scipy import stats

def get_spikes(
    csv_path,
    sampling_frequency,
):
    """

    Parameters
    ----------
    csv_path: string
    sampling_frequency: float
        in Hertz

    Returns
    -------
    List
        Each entry is the spike time (in ticks) for a particular neuron
        The list has one entry per unit.
        Each entry is a np.ndarray[int]
    """

    (spike_times, spike_labels, max_channel) = read_spikesorter_csv(csv_path, sampling_frequency)

    neuron_ids = np.unique(spike_labels)

    spike_times_for_each_neuron = []

    for neuron_id in neuron_ids:
        spike_times_for_each_neuron.append(spike_times[spike_labels == neuron_id])

    return spike_times_for_each_neuron


def read_spikesorter_csv(
    csv_path,
    sampling_frequency,
):
    """
    Sorted spike times from spikesorter (Swindale lab) are exported
    to a 3 column CSV file
    First Column: spike times (in seconds)
    Second Column: unit ID
    Third Column: Channel ID (largest SNR)



    Parameters
    ----------
    csv_path: string
    sampling_frequency: float
        in Hertz

    Returns
    -------
    np.ndarray[int]:
        spike times in units of ticks (1 index per sampling period)
    np.ndarray[int]
        unit ID
    np.ndarray[int]:
        maximum channel ID for corresponding unit
    """

    unit_list = []

    csvfile = open(csv_path, newline="")
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        unit_list.append(row)

    unit_list = unit_list[2:]  # Remove header

    spike_times = np.zeros(len(unit_list), dtype=int)

    spike_labels = np.zeros(len(unit_list), dtype=int)
    max_channel = np.zeros(len(unit_list), dtype=int)

    for i in range(len(unit_list)):
        spike_times[i] = float(unit_list[i][0]) * sampling_frequency
        spike_labels[i] = unit_list[i][1]
        max_channel[i] = unit_list[i][2]

    return spike_times, spike_labels, max_channel


def read_best_channel(csv_path, sampling_frequency):
    """

    Returns the highest SNR channel for each unique unit in the csv file

    Parameters
    ----------
    csv_path: string
    sampling_frequency: float

    Returns
    -------
    list:
        highest SNR channel for each unique unit in the csv
    """

    spike_times, spike_labels, max_channel = read_spikesorter_csv(
        csv_path,
        sampling_frequency,
    )

    neuron_ids = np.unique(spike_labels)

    best_channel = []

    for neuron_id in neuron_ids:

        best_channel.append(stats.mode(max_channel[spike_labels == neuron_id])[0][0])

    return best_channel