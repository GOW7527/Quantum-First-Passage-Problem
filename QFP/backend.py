import numpy as np
import multiprocessing as mp


# Return Problem
def return_amplitude(model):
    """
    This function calculates the Loschmidt amplitude at time tau,2tau,3tau,...,N*tau, where N is the number of time steps.
    """
    time_array = np.arange(1, int(model.N) + 1, 1)
    if model.tau is not None:
        time_array = time_array * model.tau
    eigen_matrix = np.outer(time_array, model.E)
    time_evolution = np.exp(-1j * eigen_matrix, dtype=np.complex_)
    psi_0 = np.abs(np.transpose(model.V) @ model.psi_0) ** 2
    loschmidt_amplitude = time_evolution @ psi_0
    return loschmidt_amplitude


# Arrival Problem
def arrival_amplitude(model):
    """
    This function calculates the amplitude of the probability of arrival at time tau, 2tau, 3tau,...,N*tau, where N is the number of time steps.
    """
    pass


def first_detection_amplitude_calculator(amplitudes, parallel=False):
    """
    This function calculates the first detection amplitude, phi_n, either from multiple sets of amplitudes or just a single one.
    """
    if amplitudes.ndim == 1 and parallel == False:
        n = amplitudes.shape[0]
        phi = np.zeros(n, dtype=np.complex_)
        phi[0] = amplitudes[0]
        for i in range(1, n):
            inverse = amplitudes[:i][::-1]
            phi[i] = amplitudes[i] - np.sum(phi[:i] * inverse)
        return phi
    elif amplitudes.ndim == 2 and parallel == False:
        n = amplitudes.shape[1]
        phi = np.zeros(n, dtype=np.complex_)
        phi[:, 0] = amplitudes[:, 0]
        for i in range(1, n):
            inverse = amplitudes[:, :i][:, ::-1]
            phi[:, i] = amplitudes[:, i] - np.sum(phi[:, :i] * inverse, axis=1)
        return phi
    elif amplitudes.ndim == 2 and parallel == True:
        with mp.Pool() as pool:
            results = pool.starmap(
                first_detection_amplitude_calculator,
                [
                    (
                        amplitudes[i],
                        False,
                    )
                    for i in range(amplitudes.shape[0])
                ],
            )
        return np.array(results, dtype=np.complex_)
