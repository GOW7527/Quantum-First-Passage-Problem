from QFP.backend import *
import multiprocessing as mp
import numpy as np


def probability_amplitude(model):
    if model.name == "return_problem":
        return return_amplitude(model)
    elif model.name == "arrival_problem":
        return arrival_amplitude(model)
    elif "multiple" in model.name:
        with mp.Pool() as pool:
            results = pool.map(probability_amplitude, model.models)
        Amplitudes = np.array(results, dtype=np.complex_)
        return Amplitudes
    else:
        raise ValueError("Model is not well defined")


def first_detection_amplitude(model=None, amplitudes=None):
    """
    This function calculates the first detection amplitude, phi_n, either by feeding a Hamiltonian model or a list of Amplitudes.
    If the function receives the Amplitudes, it can calculate the first detection amplitude from multiple sets of Loschmidt Amplitudes. A single row of Amplitudes is considered as a single set.
    """
    if model is None and amplitudes is None:
        raise ValueError("Either model or loschmidt_amplitude must be given")
    if model is not None and amplitudes is not None:
        raise ValueError("Only model or loschmidt_amplitude must be given")
    if amplitudes is not None and model is None:
        return first_detection_amplitude_calculator(amplitudes)
    if amplitudes is None and model is not None:
        amplitudes = probability_amplitude(model)
        return first_detection_amplitude_calculator(amplitudes)


def first_detection_probability(model=None, amplitudes=None):
    """
    This function calculates the first detection probability, F_n, either by feeding a Hamiltonian model or the Loschmidt Amplitude.
    If the function receives the Loschmidt_amplitude, it can calculate the first detection probability from multiple sets of Loschmidt Amplitudes. A single row of Loschmidt Amplitudes is considered as a single set.
    First_detection_probability calls first_detection_amplitude and then squares the result.
    """
    if model is None and amplitudes is None:
        raise ValueError("Either model or loschmidt_amplitude must be given")
    if model is not None and amplitudes is not None:
        raise ValueError("Only model or loschmidt_amplitude must be given")
    phi = first_detection_amplitude(model=model, amplitudes=amplitudes)
    assert phi is not None
    return np.abs(phi) ** 2
