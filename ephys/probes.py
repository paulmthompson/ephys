import numpy as np
import probeinterface
from probeinterface import Probe


def get_poly2_probe():
    n = 32
    positions = np.zeros((n, 2))

    height = 300
    for i in range(10):
        positions[i] = -21.65, height
        height += 50

    height = 250
    for i in range(10, 16):
        positions[i] = -21.65, height
        height -= 50

    height = 25
    for i in range(16, 22):
        positions[i] = 21.65, height
        height += 50

    height = 25 + 15 * 50
    for i in range(22, 32):
        positions[i] = 21.65, height
        height -= 50

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7.5})

    probeinterface.wiring.pathways["A32>RHD2132"] = [
        30,
        26,
        21,
        17,
        27,
        22,
        20,
        25,
        28,
        23,
        19,
        24,
        29,
        18,
        31,
        16,
        0,
        15,
        2,
        13,
        8,
        9,
        7,
        1,
        6,
        14,
        10,
        11,
        5,
        12,
        4,
        3,
    ]
    probe.wiring_to_device("A32>RHD2132")
    return probe


def get_poly3_probe():
    n = 32
    positions = np.zeros((n, 2))

    height = 12.5
    for i in range(10):
        positions[i] = -18, height
        height += 25

    height = 0
    for i in range(10, 16):
        positions[i] = 0, height
        height += 50

    height -= 25
    for i in range(16, 22):
        positions[i] = 0, height
        height -= 50

    height = 237.5
    for i in range(22, 32):
        positions[i] = 18, height
        height -= 25

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 7.5})

    probeinterface.wiring.pathways["A32>RHD2132"] = [
        30,
        26,
        21,
        17,
        27,
        22,
        20,
        25,
        28,
        23,
        19,
        24,
        29,
        18,
        31,
        16,
        0,
        15,
        2,
        13,
        8,
        9,
        7,
        1,
        6,
        14,
        10,
        11,
        5,
        12,
        4,
        3,
    ]
    probe.wiring_to_device("A32>RHD2132")
    return probe


def get_probe(probe_type):
    if probe_type.lower() == "poly2":
        return get_poly2_probe()
    elif probe_type.lower() == "poly3":
        return get_poly3_probe()
    else:
        raise ValueError(f"Unknown probe type: '{probe_type}'. Supported types: 'poly2', 'poly3'")
