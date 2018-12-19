import sys
import datetime
from os import path,mkdir

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib


############################################
fil_name = sys.argv[1]
fil = open(fil_name, 'r')
params = {}
for line in fil:
    vals = line.split(",")
    if vals[0] in ["N_row","N_col"]:
        params[vals[0]] = int(vals[1])
    elif vals[0] in ["dirc"]:
        params[vals[0]] = vals[1][:-1]
    else:
        params[vals[0]] = float(vals[1])

dirc = params['dirc']
if not path.isdir(dirc):
    mkdir(dirc)
    
N_row = params["N_row"]
N_col = params["N_row"]
Omega = params["Omega"]
J=params['J']
h=params['h']
g=params['g']
delta_ini=params['delta_ini']

dirc = dirc+'/h'+str(int(1000*h))+'/'
if not path.isdir(dirc):
    mkdir(dirc)

T = 2*np.pi/Omega
Jz_row = np.array([[J]*(N_col)]*(N_row+1))
Jz_col = np.array([[J]*(N_col+1)]*(N_row))
hz0 = np.array([[g]*N_col]*N_row)
hx = np.array([[h]*N_col]*N_row)
Ttotal=params['Ttotal']
nT=int(Ttotal/T)

theta_random = 2*np.pi*np.random.random(size=[N_row+2,N_col+2])
r_random = delta_ini*np.random.random(size=[N_row+2,N_col+2])
#Sx = np.zeros(shape=[N_row+2,N_col+2],dtype=np.float)
#Sy = np.zeros(shape=[N_row+2,N_col+2],dtype=np.float)
Sx = r_random*np.cos(theta_random)
Sy = r_random*np.sin(theta_random)
print np.max(np.abs(r_random))
Sz = np.sqrt(1.-Sx**2.-Sy**2.)**0.5*np.array([[(-1.)**(i+j) for i in range(N_col+2)] for j in range(N_row+2)])

E=np.zeros(nT+1)
E[0]=0.5*(np.sum(hx*Sx[1:-1,1:-1])+np.sum(hz0*Sz[1:-1,1:-1])+np.sum(Jz_row[1:-1,:]*Sz[2:-1,1:-1]*Sz[1:-2,1:-1])+np.sum(Jz_col[:,1:-1]*Sz[1:-1,2:-1]*Sz[1:-1,1:-2]))
for i in range(nT):
    theta = (hz0+Jz_row[1:,:]*Sz[2:,1:-1]+Jz_row[:-1,:]*Sz[:-2,1:-1]+Jz_col[:,1:]*Sz[1:-1,2:]+Jz_col[:,:-1]*Sz[1:-1,:-2])*T/2
    cos,sin=np.cos(theta),np.sin(theta)
    Sx[1:-1,1:-1],Sy[1:-1,1:-1]=cos*Sx[1:-1,1:-1]-sin*Sy[1:-1,1:-1],sin*Sx[1:-1,1:-1]+cos*Sy[1:-1,1:-1]
    #Sy[1:-1,1:-1],Sx[1:-1,1:-1]=cos*Sy[1:-1,1:-1]+sin*Sx[1:-1,1:-1],-sin*Sy[1:-1,1:-1]+cos*Sx[1:-1,1:-1]

    phi = hx*T/2
    cos,sin=np.cos(phi),np.sin(phi)
    Sy[1:-1,1:-1],Sz[1:-1,1:-1]=cos*Sy[1:-1,1:-1]-sin*Sz[1:-1,1:-1],sin*Sy[1:-1,1:-1]+cos*Sz[1:-1,1:-1]
    Jz_col[:,1:-1]*Sz[1:-1,2:-1]*Sz[1:-1,1:-2]
    E[i+1]=0.5*(np.sum(hx*Sx[1:-1,1:-1])+np.sum(hz0*Sz[1:-1,1:-1])+np.sum(Jz_row[1:-1,:]*Sz[2:-1,1:-1]*Sz[1:-2,1:-1])+np.sum(Jz_col[:,1:-1]*Sz[1:-1,2:-1]*Sz[1:-1,1:-2]))
    
np.savetxt(dirc+"E_N"+str(N_row)+"_"+str(N_col)+"_Omega"+str(Omega)+".csv",E/(N_row*N_col),delimiter=',')