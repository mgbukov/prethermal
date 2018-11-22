import ffht
import timeit

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel

import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=4,suppress=True)

############################################

L=16 # lattice sites
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
static_x_diag=[['z',transverse_field]]

# basis
basis=spin_basis_1d(L=L,pauli=True) # uses pauli matrices, NOT spin-1/2 operators

# calculate Hamiltonians
no_checks=dict(check_symm=False,check_pcon=False,check_herm=False)
Hz=hamiltonian(static_z,[],basis=basis,dtype=np.float64,**no_checks)
Hx=hamiltonian(static_x,[],basis=basis,dtype=np.float64,**no_checks)
H_ave=0.5*(Hz+Hx)

Hx_diag=hamiltonian(static_x_diag,[],basis=basis,dtype=np.float64,**no_checks).diagonal()
Hz_diag=Hz.diagonal()

# define initial state
index_i=basis.index('0'*L)
psi_0=np.zeros(basis.Ns,dtype=np.complex128)
psi_0[index_i]=1.0

# preallocate matrix exponentials exp(a*H)
expHz = expm_multiply_parallel(Hz.tocsr(),a=-1j*0.5*T)
expHx = expm_multiply_parallel(Hx.tocsr(),a=-1j*0.5*T)

expHz_diagonal=np.exp(-1j*0.5*T*Hz_diag)
expHx_diagonal=np.exp(-1j*0.5*T*Hx_diag)

cosHz_diagonal = +np.cos(0.5*T*Hz_diag)
sinHz_diagonal = -np.sin(0.5*T*Hz_diag)
cosHx_diagonal = +np.cos(0.5*T*Hx_diag)
sinHx_diagonal = -np.sin(0.5*T*Hx_diag)


# evolve state
nT=100
subsys=range(L//2)
energy_t=np.zeros((nT,),dtype=np.float64)
sent_t=np.zeros((nT,),dtype=np.float64)


#"""
psi=psi_0.copy()
work_array=np.zeros((2*len(psi),), dtype=psi.dtype)

t1 = timeit.default_timer()
for j in range(nT):
	# evolve state

	expHx.dot(psi,work_array=work_array,overwrite_v=True) # applies exp(-iTHx/2)
	expHz.dot(psi,work_array=work_array,overwrite_v=True) # applies exp(-iTHx/2)

	# measure state
	sent_t[j]=basis.ent_entropy(psi,sub_sys_A=subsys)['Sent_A'] # entanglement entropy per site
	energy_t[j]=H_ave.expt_value(psi).real/L # energy density

	#print('finished evolving time step {0:d}'.format(j))	

t2 = timeit.default_timer()
print('matrix_exp evolution took {0:f} secs'.format(t2-t1))
print('matrix_exp final entropy is {0:f}'.format(basis.ent_entropy(psi,sub_sys_A=subsys)['Sent_A']) )
#"""





psi_FFHT=np.zeros((2*basis.Ns,),dtype=np.float64)
psi_FFHT[:basis.Ns]=psi_0.real.copy()
psi_FFHT[basis.Ns:]=psi_0.imag.copy()

psi_FFHT_cpx=np.zeros((basis.Ns,),dtype=np.complex128)

norm=np.sqrt(2)**L
a=np.zeros_like(psi_0)

t1 = timeit.default_timer()
for j in range(nT):
	# evolve state using FFHT

	# apply FHT 
	ffht.fht(psi_FFHT[:basis.Ns])
	ffht.fht(psi_FFHT[basis.Ns:])
	psi_FFHT/=norm

	# apply Ux

	''' (a+ib)(c+id)=ac-bd + i(ad+bc)
	z=cosHx_diagonal*psi_FFHT[:basis.Ns] - sinHx_diagonal*psi_FFHT[basis.Ns:]
	psi_FFHT[basis.Ns:] = sinHx_diagonal*psi_FFHT[:basis.Ns] + cosHx_diagonal*psi_FFHT[basis.Ns:]
	psi_FFHT[:basis.Ns]=z[:]
	'''
	a=sinHx_diagonal*psi_FFHT[basis.Ns:]
	psi_FFHT[basis.Ns:]*=cosHx_diagonal
	psi_FFHT[basis.Ns:]+=sinHx_diagonal*psi_FFHT[:basis.Ns]
	psi_FFHT[:basis.Ns]*=cosHx_diagonal
	psi_FFHT[:basis.Ns]-=a
	
	# apply FHT
	ffht.fht(psi_FFHT[:basis.Ns])
	ffht.fht(psi_FFHT[basis.Ns:])
	psi_FFHT/=norm

	
	# apply Uz
	a=sinHz_diagonal*psi_FFHT[basis.Ns:]
	psi_FFHT[basis.Ns:]*=cosHz_diagonal
	psi_FFHT[basis.Ns:]+=sinHz_diagonal*psi_FFHT[:basis.Ns]
	psi_FFHT[:basis.Ns]*=cosHz_diagonal
	psi_FFHT[:basis.Ns]-=a

	# complexify state
	psi_FFHT_cpx.real=psi_FFHT[:basis.Ns]
	psi_FFHT_cpx.imag=psi_FFHT[basis.Ns:]

	# measure state
	sent_t[j]=basis.ent_entropy(psi_FFHT_cpx,sub_sys_A=subsys)['Sent_A'] # entanglement entropy per site
	energy_t[j]=H_ave.expt_value(psi_FFHT_cpx).real/L # energy density


	#print('finished evolving time step {0:d}'.format(j))

t2 = timeit.default_timer()
# construct complex psi
psi_FFHT = psi_FFHT[:basis.Ns] + 1j*psi_FFHT[basis.Ns:]


print('FFHT evolution took {0:f} secs'.format(t2-t1))
print('FFHT final entropy is {0:f}'.format(basis.ent_entropy(psi_FFHT,sub_sys_A=subsys)['Sent_A']) )


print('L2 error between two solutions is}', np.linalg.norm(psi_FFHT-psi) )




