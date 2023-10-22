import os 
os.environ["OMP_NUM_THREADS"]="1"
from backend import *
size=int(5e3)
iterations=100
F=np.zeros((iterations,size),dtype=np.float64)
a=1-np.sqrt(3)
b=1+np.sqrt(3)
psi_0=np.zeros(size,dtype=np.complex_)
psi_0[size//2]=1
for i in range(iterations):
    diagonal=np.random.normal(0,scale=1.0,size=size,dtype=np.float64) 
    off_diagonal=np.random.uniform(a,b,size-1)
    eigenvalues,eigenvectors=spectral_decomposition(off_diagonal,diagonal)
    amplitude=loschmidt_amplitude(eigenvalues,eigenvectors,psi_0)
    phi=first_detection_amplitude(amplitude)
    F[i]=np.abs(phi)**2
np.save('F-uniform_odd_gaussian_d.npy',F)


