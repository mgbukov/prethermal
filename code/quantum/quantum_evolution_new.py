import sys
import datetime
from os import path,mkdir
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg
import timeit
sys.path += ["/mnt/d/ED/QuSpin"]
sys.path += ["/mnt/d/ED/FFHT-master"]
sys.path += ["/n/lukin_lab2/Users/nyao/Bingtian/ED/QuSpin"]
sys.path += ["/n/lukin_lab2/Users/nyao/Bingtian/ED/FFHT-master"]

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.evolution import expm_multiply_parallel
import ffht

np.set_printoptions(precision=4,suppress=True)

############################################
fil_name = sys.argv[1]
fil = open(fil_name, 'r')
params = {}
for line in fil:
    vals = line.split(",")
    if vals[0] in ["L","PBC"]:
        params[vals[0]] = int(vals[1])
    elif vals[0] in ["dirc"]:
        params[vals[0]] = vals[1][:-1]
    else:
        params[vals[0]] = float(vals[1])

dirc = params['dirc']
if not path.isdir(dirc):
    mkdir(dirc)

L=params['L'] # lattice sites
PBC=params['PBC'] # periodic BC

# model params, Hamiltonian in terms of Pauli matrices
J=params['J']
h=params['h'] # transverse field
g=params['g'] # parallel field
delta_g=params['delta_g'] # disorder of parallel field to break the symmetry

Omega=params['Omega'] # driving frequency
Ttotal=params['Ttotal'] # total evolution time
T=2.*np.pi/Omega # driving period

dirc = dirc+'/h'+str(int(h*1000))+'/'
if not path.isdir(dirc):
    mkdir(dirc)

# Hamiltonian
g_disorder = (2*np.random.random(size=L)-1.)*g*delta_g
g_set = np.array(g+g_disorder)
z_set=np.zeros(shape=(L,2**L))   # store sigma^z_i vectors
sigma_z = np.array([1.,-1.])
for sitenum in range(L):
    z_set[sitenum] = np.kron(np.kron(np.ones(2**sitenum),sigma_z),np.ones(2**(L-sitenum-1)))

Hz_static = np.zeros(2**L)
sigma_zz = np.kron(np.array([1.,-1.]),np.array([1.,-1.]))
for sitenum in range(L-1):
    Hz_static += J*np.kron(np.kron(np.ones(2**sitenum),sigma_zz),np.ones(2**(L-sitenum-2)))
if PBC:
    Hz_static += J*np.kron(np.kron(sigma_z,np.ones(2**(L-2))),sigma_z)

Hz_static += np.dot(g_set,z_set)
cosHz_static=np.cos(0.5*T*Hz_static)
sinHz_static=np.sin(0.5*T*Hz_static)

Hx_diag = np.dot(np.array([h]*L),z_set)
cosHx_diag=np.cos(0.5*T*Hx_diag)
sinHx_diag=np.sin(0.5*T*Hx_diag)

g_dynamic = np.zeros(shape=L)
Hz_dynamic = np.zeros(shape=2**L)
cosHz_dynamic = np.ones(shape=2**L)
sinHz_dynamic = np.zeros(shape=2**L)
# basis
basis=spin_basis_1d(L=L,pauli=True) # uses pauli matrices, NOT spin-1/2 operators
# define initial state
index_i=basis.index('10'*(L/2))
psi0=np.zeros(basis.Ns,dtype=np.complex128)
psi0[index_i]=1.0
psi=np.zeros((2*basis.Ns,),dtype=np.float64)
psi[:basis.Ns]=psi0.real.copy()
psi[basis.Ns:]=psi0.imag.copy()

# evolve state
nT=int(Ttotal/T)
subsys=range(L//2)
energy=np.zeros((nT,),dtype=np.float64)
#sent=np.zeros((nT,),dtype=np.float64)

norm=basis.Ns #2**L
a=np.zeros_like(psi) #cache

t1 = timeit.default_timer()
for j in range(nT):
	#'''
	# evolve state using FFHT
	# apply Uz_dynamic
	g_dynamic = np.array([0.0]*L)   #evaluate local field
	#g_dynamic = (2*np.random.random(size=L)-1.)*g   #evaluate local field
	Hz_dynamic = np.dot(g_dynamic,z_set)
	cosHz_dynamic = np.cos(0.5*T*Hz_dynamic)
	sinHz_dynamic = np.sin(0.5*T*Hz_dynamic)
	a=sinHz_dynamic*psi[basis.Ns:]
	psi[basis.Ns:]*=cosHz_dynamic
	psi[basis.Ns:]+=sinHz_dynamic*psi[:basis.Ns]
	psi[:basis.Ns]*=cosHz_dynamic
	psi[:basis.Ns]-=a

	
	# apply Uz_static
	a=sinHz_static*psi[basis.Ns:]
	psi[basis.Ns:]*=cosHz_static
	psi[basis.Ns:]+=sinHz_static*psi[:basis.Ns]
	psi[:basis.Ns]*=cosHz_static
	psi[:basis.Ns]-=a
	'''
########################################
	# evolve state using FFHT
	# apply Uz
	g_dynamic = np.array([0.0]*L)   #evaluate local field
	g_dynamic = (2*np.random.random(size=L)-1.)*g   #evaluate local field
	Hz = np.dot(g_dynamic,z_set)
	Hz += Hz_static
	cosHz = np.cos(0.5*T*Hz)
	sinHz = np.sin(0.5*T*Hz)
	a=sinHz*psi[basis.Ns:]
	psi[basis.Ns:]*=cosHz
	psi[basis.Ns:]+=sinHz*psi[:basis.Ns]
	psi[:basis.Ns]*=cosHz
	psi[:basis.Ns]-=a
	#'''
########################################
	# apply FHT
	ffht.fht(psi[:basis.Ns])
	ffht.fht(psi[basis.Ns:])
	psi/=norm

	# apply Ux

	a=sinHx_diag*psi[basis.Ns:]
	psi[basis.Ns:]*=cosHx_diag
	psi[basis.Ns:]+=sinHx_diag*psi[:basis.Ns]
	psi[:basis.Ns]*=cosHx_diag
	psi[:basis.Ns]-=a

	energy[j]=0.5*np.dot((psi[:basis.Ns]**2+psi[basis.Ns:]**2),Hx_diag)*norm/L # energy density

	# apply FHT
	ffht.fht(psi[:basis.Ns])
	ffht.fht(psi[basis.Ns:])
	#psi_comp/=norm

	# measure state
	#sent_t[j]=basis.ent_entropy(psi_FFHT_cpx,sub_sys_A=subsys)['Sent_A'] # entanglement entropy per site
	energy[j]+=0.5*np.dot((psi[:basis.Ns]**2+psi[basis.Ns:]**2),Hz_static)/L # energy density

	print('finished evolving time step {0:d}'.format(j))

t2 = timeit.default_timer()
# construct complex psi
psi_full = psi[:basis.Ns] + 1j*psi[basis.Ns:]


print('FFHT evolution took {0:f} secs'.format(t2-t1))


