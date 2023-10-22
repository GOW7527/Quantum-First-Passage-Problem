import os 
os.environ["OMP_NUM_THREADS"]="1"
from backend import *
size=int(5e3)
iterations=100
F=np.zeros((iterations,size),dtype=np.float64)
W=np.sqrt(12)
psi_0=np.zeros(size,dtype=np.complex_)
psi_0[size//2]=1
for i in range(iterations):
    diagonal=np.random.uniform(-W/2,W/2,size)
    off_diagonal=np.random.normal(1,scale=1.0,size=size-1,dtype=np.float64) 
    eigenvalues,eigenvectors=spectral_decomposition(off_diagonal,diagonal)
    amplitude=loschmidt_amplitude(eigenvalues,eigenvectors,psi_0)
    phi=first_detection_amplitude(amplitude)
    F[i]=np.abs(phi)**2
np.save('F-gaussian_odd_uniform_d.npy',F)


