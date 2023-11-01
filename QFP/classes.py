import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.linalg import ishermitian
import multiprocessing as mp

def is_tridiagonal(model):
    n=len(model.array)
    for i in range(n):
        for j in range(n):
            if abs(i-j)>1 and model.array[i][j]!=0:
                return False
    return True

class return_problem:
    def __init__(self,hamiltonian,psi_0,tau=None,N=1e4):
        '''
        psi_0 is the initial state of the system
        Default tau=1 and calculates up to N=1e4
        N*tau is the maximum time of the simulation
        '''
        self.name='return_problem'
        if ishermitian(hamiltonian)==False:
            raise ValueError("Hamiltonian is not hermitian")        #The Hamiltonian of the system must be Hermitian
        self.array=hamiltonian
        self.psi_0=psi_0
        self.tau=tau
        self.N=int(N)
        self.tridiagonal=is_tridiagonal(self)

        if self.tridiagonal:                                #If the Hamiltonian is tridiagonal, we can use the faster exact diagonalization procedure
            D=self.array.diagonal()
            off_D=self.array.diagonal(1)
            self.E,self.V=eigh_tridiagonal(D,off_D)
        else:
            self.E,self.V=np.linalg.eigh(self.array)

class arrival_problem:
    def __init__(self,hamiltonian,psi_0,psi_T,tau=None,N=1e4):
        self.name='arrival_problem'
        '''
        psi_0 is the initial state of the system
        psi_T is the target state of the system
        Default tau=1 and calculates up to N=1e4
        N*tau is the maximum time of the simulation
        '''
        if ishermitian(hamiltonian)==False:
            raise ValueError("Hamiltonian is not hermitian")        #The Hamiltonian of the system must be Hermitian
        self.array=hamiltonian
        self.psi_0=psi_0
        self.psi_T=psi_T
        self.tau=tau
        self.N=int(N)
        self.tridiagonal=is_tridiagonal(self)
        if self.tridiagonal:                                #If the Hamiltonian is tridiagonal, we can use the faster exact diagonalization procedure
            D=self.array.diagonal()
            off_D=self.array.diagonal(1)
            self.E,self.V=eigh_tridiagonal(D,off_D)
        else:
            self.E,self.V=np.linalg.eigh(self.array)

class multiple_return_problems:
    '''
    This is suited for the case we study disordered systems and need to perform an average over many realizations of the disorder. Or interested in comparing multiple Hamiltonians at the same time
    '''
    def __init__(self,H_list,psi_0,tau=None,N=1e4,parallel=False):
        '''
        H_list: list of Hamiltonians
        psi_0: initial state
        tau: time step
        N: number of steps
        '''
        self.name='multiple_return_problems'
        self.n=len(H_list)
        self.N=int(N)
        if parallel:
            with mp.Pool() as pool:
                self.models=pool.starmap(return_problem,[(H_list[i],psi_0,tau,N) for i in range(self.n)])
        else:
            self.models=[None]*self.n
            for i in range(self.n):
                self.models[i]=return_problem(H_list[i],psi_0,tau,N)

class multiple_arrival_problems:
    '''
    This is suited for the case we study disordered systems and need to perform an average over many realizations of the disorder. Or interested in comparing multiple Hamiltonians at the same time
    '''
    def __init__(self,H_list,psi_0,psi_T,tau=None,N=1e4):
        '''
        H_list: list of Hamiltonians
        psi_0: initial state
        psi_T: target state
        tau: time step
        N: number of steps
        '''
        self.name='multiple_arrival_problems'
        self.n=len(H_list)
        self.models=[None]*self.n
        for i in range(self.n):
            self.models[i]=arrival_problem(H_list[i],psi_0,psi_T,tau,N)


