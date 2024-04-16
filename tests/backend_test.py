from QFP.backend import *
import numpy as np
from scipy.special import j0
import matplotlib.pyplot as plt


def test_return_amplitude():
    # Tau = [0.1,0.5,1.0,2.0,5.0]
    dt = 0.1
    T = 100
    n = int(T/dt)
    x = np.arange(0, n)
    time = x * dt
    true_amplitude = j0(2 * time)
    plt.plot(time, true_amplitude)
    plt.show()


test_return_amplitude()
