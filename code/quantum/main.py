from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4,suppress=True)

############################################

L=18 # lattice sites
PBC=0 # periodic BC

# model params, Hamiltonian in terms of Pauli matrices
J=1.0
h=0.809 # transverse field
g=0.9045 # parallel field

Omega=5.0 # driving frequency
T=2*np.pi/Omega # driving period

# site-coupling lists
if PBC:
	nn_int=[[1.0,j,(j+1)%L] for j in range(L)]
else:
	nn_int=[[1.0,j,j+1] for j in range(L-1)]
parallel_field=[[g,j] for j in range(L)]
transverse_field=[[h,j] for j in range(L)]


# opstr lists
static_z=[['zz',nn_int],['z',parallel_field]]
static_x=[['x',transverse_field]]

# basis
basis=spin_basis_1d(L=L,pauli=True) # uses pauli matrices, NOT spin-1/2 operators

# calculate Hamiltonians
no_checks=dict(check_symm=False,check_pcon=False,check_herm=False)
Hz=hamiltonian(static_z,[],basis=basis,dtype=np.float64,**no_checks)
Hx=hamiltonian(static_x,[],basis=basis,dtype=np.float64,**no_checks)
H_ave=0.5*(Hz+Hx)

# define initial state
index_i=basis.index('0'*L)
psi=np.zeros(basis.Ns,dtype=np.complex128)
psi[index_i]=1.0

# preallocate matrix exponentials exp(a*H)
expHz = expm_multiply_parallel(Hz.tocsr(),a=-1j*0.5*T)
expHx = expm_multiply_parallel(Hx.tocsr(),a=-1j*0.5*T)

# evolve state
nT=80
subsys=range(L//2)

# preallocate observables
energy_t=np.zeros((nT,),dtype=np.float64)
sent_t=np.zeros((nT,),dtype=np.float64)

for j in range(nT):
	# evolve state
	psi=expHx.dot(psi) # applies exp(-iTHx/2)
	psi=expHz.dot(psi) # applies exp(-iTHx/2)
	# measure state
	sent_t[j]=basis.ent_entropy(psi,sub_sys_A=subsys)['Sent_A'] # entanglement entropy per site
	energy_t[j]=H_ave.expt_value(psi)/L # energy density

	print('finished evolving time step {0:d}'.format(j))	


# plot data
plt.plot(range(nT),sent_t,'--r.')
plt.plot(range(nT),energy_t,'--b.')
plt.show()





