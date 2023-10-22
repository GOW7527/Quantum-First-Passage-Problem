import numpy as np
import scipy.linalg as la
def loschmidt_amplitude(eigenvalues,eigenvectors,psi_0,time_array=None):
    if time_array is None:
        time_array=np.arange(1,int(1e4)+1,1)
    eigen_matrix=np.outer(time_array,eigenvalues)
    time_evolution=np.exp(-1j*eigen_matrix,dtype=np.complex_)
    psi_0=np.abs(eigenvectors.T@psi_0)**2
    loschmidt_amplitude=time_evolution@psi_0
    return loschmidt_amplitude
def first_detection_amplitude(loschmidt_amplitude):
    if loschmidt_amplitude.ndim==1:
        n=loschmidt_amplitude.shape[0]
        phi=np.zeros_like(loschmidt_amplitude,dtype=np.complex_)
        phi[0]=loschmidt_amplitude[0]
        for i in range(1,n):
            inverse=loschmidt_amplitude[:i][::-1]
            phi[i]=loschmidt_amplitude[i]-np.sum(phi[:i]*inverse)
        return phi
    elif loschmidt_amplitude.ndim==2:
        n=loschmidt_amplitude.shape[1]
        phi=np.zeros_like(loschmidt_amplitude,dtype=np.complex_)
        phi[:,0]=loschmidt_amplitude[:,0]
        for i in range(1,n):
            inverse=loschmidt_amplitude[:,:i][:,::-1]
            phi[:,i]=loschmidt_amplitude[:,i]-np.sum(phi[:,:i]*inverse,axis=1)
        return phi
def spectral_decomposition(off_diagonal_term,diagonal_term):
    eigenvalues,eigenvectors=la.eigh_tridiagonal(diagonal_term,off_diagonal_term)
    return eigenvalues,eigenvectors
if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import j0
    x=np.arange(1,int(1e4)+1,1,dtype=np.float64)
    time=x*np.pi/2
    amplitude=j0(2*time)
    phi=first_detection_amplitude(amplitude)
    F=np.abs(phi)**2
    plt.loglog(x,F)
    y=0.25*x**(-3)  
    plt.loglog(x,y)
    plt.show()


