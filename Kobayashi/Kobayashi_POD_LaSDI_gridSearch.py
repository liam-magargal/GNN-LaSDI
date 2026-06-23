import numpy as np
import sys
import torch

from typing import Callable, Optional, Tuple



def getLasdiOperators(lamb,D,R):
    Gamma = np.zeros((1+N_lat+int((N_lat+1)*N_lat/2),1+N_lat+int((N_lat+1)*N_lat/2)))
    Gamma[0,0] = 10**lamb[0]
    Gamma[1:N_lat+1,1:N_lat+1] = 10**lamb[0]*np.eye(N_lat)
    Gamma[N_lat+1:,N_lat+1:] = 10**lamb[1]*np.eye(int((N_lat+1)*N_lat/2))
    
    
    D = D.T
    
    LHS = D.T@D + Gamma.T@Gamma
    RHS = D.T@R.T
    Ot = np.linalg.solve(LHS,RHS)
    O = Ot.T
    C = O[:,0]
    A = O[:,1:N_lat+1]
    F = O[:,N_lat+1:]
    
    return C, A, F


def getLasdiSolBE(C,A,F,x_init,N_lat,Nt,dt):
    tol = 1e-6
    x_hat_opInf = np.zeros((N_lat,Nt))
    x_hat_opInf[:,0] = x_init
    
    for t in range(Nt-1):
        x_hat_curr = x_hat_opInf[:,t]
        x_hat_next = x_hat_opInf[:,t]
        
        iterCount = 0
        while True:
            x_hat_sq = np.zeros((int(N_lat*(N_lat+1)/2)))
            dsq_dxhat = np.zeros((int(N_lat*(N_lat+1)/2),N_lat))
            count = 0
            for j in range(N_lat):
                for k in range(j+1):
                    x_hat_sq[count] = x_hat_next[j]*x_hat_next[k]
                    dsq_dxhat[count,j] += x_hat_next[k]
                    dsq_dxhat[count,k] += x_hat_next[j]
                    count += 1
            
            r = x_hat_next - x_hat_curr - dt*(C - A@x_hat_next - F@x_hat_sq)
            J = np.eye(N_lat) - dt*(A - F@dsq_dxhat)
            
            if np.linalg.norm(r)>tol:
                x_hat_next = x_hat_next - np.linalg.solve(J,r)
                iterCount += 1
            else:
                x_hat_opInf[:,t+1] = x_hat_next
            
            if iterCount>100:
                x_hat_opInf[:,t:-1] = 1e8*np.zeros(np.shape(x_hat_opInf))
                return x_hat_opInf
            
    return x_hat_opInf


def getLasdiErrorBE(lamb,D,R,x_hat_train,N_lat,Nt,dt):
    
    C, A, F = getLasdiOperators(lamb,D,R)
    
    x_hat_opInf = np.zeros((25,N_lat,Nt))
    x_hat_trainFlat = np.zeros((N_lat,25*Nt))
    x_hat_opInfFlat = np.zeros((N_lat,25*Nt))
    
    for sol in range(25):
        x_hat_opInf[sol,:,0] = x_hat_train[sol,:,0]
        for t in range(Nt-1):
            x_hat_curr = x_hat_opInf[sol,:,t]
            x_hat_next = x_hat_opInf[sol,:,t]
            
            iterCount = 0
            while True:
                x_hat_sq = np.zeros((int(N_lat*(N_lat+1)/2)))
                dsq_dxhat = np.zeros((int(N_lat*(N_lat+1)/2),N_lat))
                count = 0
                for j in range(N_lat):
                    for k in range(j+1):
                        x_hat_sq[count] = x_hat_next[j]*x_hat_next[k]
                        dsq_dxhat[count,j] += x_hat_next[k]
                        dsq_dxhat[count,k] += x_hat_next[j]
                        count += 1
                
                r = x_hat_next - x_hat_curr - dt*(C + A@x_hat_next + F@x_hat_sq)
                J = np.eye(N_lat) - dt*(A + F@dsq_dxhat)
                
                if np.linalg.norm(r)>tol:
                    x_hat_next = x_hat_next - np.linalg.solve(J,r)
                    iterCount += 1
                else:
                    x_hat_opInf[sol,:,t+1] = x_hat_next
                    break
                
                if iterCount>100:
                    error = 1e8
                    return error
                
                
        x_hat_trainFlat[:,sol*Nt:(sol+1)*Nt] = x_hat_train[sol,:,:]
        x_hat_opInfFlat[:,sol*Nt:(sol+1)*Nt] = x_hat_opInf[sol,:,:]
        
    
    
    error = np.linalg.norm(x_hat_opInfFlat-x_hat_trainFlat,'fro')/np.linalg.norm(x_hat_trainFlat,'fro')

    return error


N_lat = int(sys.argv[1])

alpha = .9
gamma = 10
a = 0.01
K = 1
tau = .0003
eps = .01
Delta = 1.

dt = .00002
Nt = 501
nSol = 1

tol = 1e-6

nStep = 160
x = np.linspace(0,1,nStep)
y = np.linspace(0,1,nStep)
dx = x[1] - x[0]


Nx = np.shape(x)[0]
Ny = Nx
N = Nx*Ny

X, Y = np.meshgrid(x, y)
coords = np.vstack([X.ravel(), Y.ravel()]).T


intInd = np.where((coords[:,0] != 0) & (coords[:,1] != 0) & (coords[:,0] != 1) & (coords[:,1] != 1))[0]
leftInd = (np.where((coords[:,0] == 0))[0]).tolist()
rightInd = (np.where((coords[:,0] == 1))[0]).tolist()
bottomInd = (np.where((coords[:,1] == 0))[0]).tolist()
topInd = (np.where((coords[:,1] == 1))[0]).tolist()

bottomInd.remove(Nx-1)
rightInd.remove(N-1)
topInd.remove(N-Nx)
leftInd.remove(0)

boundInd = bottomInd + rightInd + topInd + leftInd
boundInd = np.array(boundInd)

x_hist_in = np.load('AllenCahn_x_hist_imp_May5_full.npy')


x_hist_in = x_hist_in[:,:,:Nt]

nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((2*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]


U = np.load('AllenCahn_U.npy')
phi = U[:,:N_lat]

x_hat_train = np.zeros((nSol,N_lat,Nt))
x_hat_dot_train = np.zeros((nSol,N_lat,Nt))
D = np.zeros((1+N_lat+int((N_lat+1)*N_lat/2), (Nt)*nSol))
R = np.zeros((N_lat, (Nt)*nSol))

# new code using higher order approximations to remove noise
for sol in range(nSol):
    x_hist = np.zeros((2*N,Nt))
    x_hist[:N,:] = x_hist_in[sol,:N,:]
    x_hist[N:,:] = x_hist_in[sol,N:,:]
    
    x_hat_train[sol,:,:] = phi.T@x_hist
    
    # for m=0, O(dt^2) forward
    R[:,sol*Nt] = (-3*x_hat_train[sol,:,0] + 4*x_hat_train[sol,:,1] - x_hat_train[sol,:,2])/(2*dt)
    
    # for m=1, O(dt^2) centered
    R[:,sol*Nt+1] = (x_hat_train[sol,:,2] - x_hat_train[sol,:,0])/(2*dt)
    
    # for m=2:end-2, O(dt^4) centered
    R[:,sol*Nt+2:(sol+1)*Nt-2] = (-x_hat_train[sol,:,4:] + 8*x_hat_train[sol,:,3:-1] - 8*x_hat_train[sol,:,1:-3] + x_hat_train[sol,:,:-4])/(12*dt)
    
    # for m=end-1, O(dt^2) centered
    R[:,(sol+1)*Nt-2] = (x_hat_train[sol,:,-1] - x_hat_train[sol,:,-3])/(2*dt)
    
    # for m=end, O(dt^2) backward
    R[:,(sol+1)*Nt-1] = (3*x_hat_train[sol,:,-1] - 4*x_hat_train[sol,:,-2] + x_hat_train[sol,:,-3])/(2*dt)
    
    
    D[0,sol*Nt:(sol+1)*Nt] = torch.ones((Nt))
    D[1:N_lat+1,sol*(Nt):(sol+1)*(Nt)] = x_hat_train[sol,:,:]
            
    count=0
    for j in range(N_lat):
        for k in range(j+1):
            D[N_lat+1+count,sol*Nt:(sol+1)*Nt] = x_hat_train[sol,j,:]*x_hat_train[sol,k,:]
            count += 1

# x_hat_train = x_hat_train.cpu().numpy()

lamb = np.zeros(2)

lamb0_all = [-8,-7,-6,-5,-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
lamb1_all = [-8,-7,-6,-5,-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

lasdiErrors = np.zeros((len(lamb0_all),len(lamb1_all)))

for i in range(len(lamb0_all)):
    print(i)
    lamb[0] = lamb0_all[i]
    
    for j in range(len(lamb1_all)):
        lamb[1] = lamb1_all[j]
        
        # lasdiErrors[i,j] = getLasdiError(lamb,D,R,x_hat_train,N_lat,Nt,dt)
        lasdiErrors[i,j] = getLasdiErrorBE(lamb,D,R,x_hat_train,N_lat,Nt,dt)
        if np.isnan(lasdiErrors[i,j]):
            lasdiErrors[i,j]=1e8
        
i_j_min = np.unravel_index(lasdiErrors.argmin(), lasdiErrors.shape)

print(lamb0_all[i_j_min[0]], lamb1_all[i_j_min[1]])
