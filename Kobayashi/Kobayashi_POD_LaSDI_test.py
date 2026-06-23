import numpy as np
import sys
import time

import torch

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


def getLasdiSol(C,A,F,x_init,N_lat,Nt,dt):
    x_hat_opInf = np.zeros((N_lat,Nt))
    x_hat_opInf[:,0] = x_init
    
    for t in range(Nt-1):
        x_hat_sq = np.zeros((int(N_lat*(N_lat+1)/2)))
        count = 0
        for j in range(N_lat):
            for k in range(j+1):
                x_hat_sq[count] = x_hat_opInf[j,t]*x_hat_opInf[k,t]
                count += 1
        
        x_hat_opInf[:,t+1] = x_hat_opInf[:,t] + dt*(C + A@x_hat_opInf[:,t] + F@x_hat_sq)
        
    return x_hat_opInf


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
            
            r = x_hat_next - x_hat_curr - dt*(C + A@x_hat_next + F@x_hat_sq)
            J = np.eye(N_lat) - dt*(A + F@dsq_dxhat)
            
            if np.linalg.norm(r)>tol:
                x_hat_next = x_hat_next - np.linalg.solve(J,r)
                iterCount += 1
            else:
                x_hat_opInf[:,t+1] = x_hat_next
                break
            
            if iterCount>100:
                x_hat_opInf[:,t:-1] = 1e8*np.zeros(np.shape(x_hat_opInf[:,t:-1]))
                return x_hat_opInf
            
    return x_hat_opInf



N_lat = int(sys.argv[1])
alpha = .9
gamma = 10
K = 1
tau = .0003
eps = .01
Delta = 1.

dt = .00002
Nt = 501
nSol = 1


latDimToCanny = [3, 5, 7]
testSolsToCanny = [0, 3, 12, 15]

lamb0_range = [2,2,2,2,2]
lamb1_range = [-8,3,3,-8,-4]


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

x_hist_in = np.load('Kobayashi_x_hist_imp_May5_full.npy')
x_hist_in_test = np.load('Kobayashi_x_hist_imp_May5_full_test.npy')

x_hist_in = x_hist_in[:,:,:Nt]

nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((2*N,nSol*Nt))


print(np.shape(x_hist))

U = np.load('Kobayashi_U.npy')
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

lamb = np.zeros(2)

lamb[0] = lamb0_range[N_lat-3]
lamb[1] = lamb1_range[N_lat-3]

C,A,F = getLasdiOperators(lamb,D,R)


nSol = np.shape(x_hist_in_test)[0]
x_hat_test = np.zeros((nSol,N_lat,Nt))

x_hist_in = torch.tensor(x_hist_in)

for sol in range(nSol):
    x_hist_test = np.zeros((2*N,Nt))
    
    x_hist_test = x_hist_in_test[sol,:,:]
    x_hat_test[sol,:,:] = phi.T@x_hist_test
    

nRep = 10

totalTime = np.zeros(nSol)
lowDimSolTime = np.zeros(nSol)
decodeTime = np.zeros(nSol)
encodeTime = np.zeros(nSol)
errors = np.zeros(nSol)
x_hist_in_test = x_hist_in_test
x_hist_in = x_hist_in

a_all = [.3875, .3875, .3875, .3875, .4625, .4625, .4625, .4625, .5375, .5375, .5375, .5375, .6125, .6125, .6125, .6125]
b_all = [.3875, .4625, .5375, .6125, .3875, .4625, .5375, .6125, .3875, .4625, .5375, .6125, .3875, .4625, .5375, .6125]



for sol in range(nSol):
    print(sol, nSol)
    for rep in range(nRep):
        ts = time.time()
        t1 = time.time()
        a = a_all[sol]
        b = b_all[sol]
    
        r0 = .075
        r = np.sqrt(np.square(coords[:,0]-a) + np.square(coords[:,1]-b))
        phi_init = .5*(1 - np.tanh((r-r0)/(2*eps*np.sqrt(2))))
        U_init = Delta*phi_init
        x_init = np.zeros((2*N))
        x_init[:N] = phi_init
        x_init[N:] = U_init
        x_hat_init = phi.T@x_init
        t2 = time.time()
        encodeTime[sol] += t2-t1
        
        t1 = time.time()
        x_hat_opInf = getLasdiSolBE(C,A,F,x_hat_init,N_lat,Nt,dt)
        t2 = time.time()
        lowDimSolTime[sol] += t2-t1
        
        t3 = time.time()
        x_approx = phi@x_hat_opInf
        t4 = time.time()
        decodeTime[sol] += t4-t3
        
        x_hist_approx = np.zeros((2*N,Nt))
        x_hist_approx[:N,:] = (x_approx[:N,:])
        x_hist_approx[N:,:] = (x_approx[N:,:])
        
        te = time.time()
        totalTime[sol] += te-ts
        
        if sol in testSolsToCanny and N_lat in latDimToCanny:
            np.save('Kobayashi_solutions/Kobayashi_POD_LaSDI_'+str(N_lat) + '_testSol_'+str(sol), x_hat_opInf)
        
        
    errors[sol] = np.linalg.norm(x_hist_approx[:,:]-x_hist_in_test[sol,:,:],'fro')/np.linalg.norm(x_hist_in_test[sol,:,:],'fro')
    
    
print(errors)
print(decodeTime/nRep)
np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_errors_',errors)
np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_lowDimSolTime_',lowDimSolTime/nRep)
np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_decodeTime_',decodeTime/nRep)
np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_encodeTime_',encodeTime/nRep)
np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_totalTime_',totalTime/nRep)
