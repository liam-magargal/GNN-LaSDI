import numpy as np
import sys
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

def getLasdiError(lamb,D,R,x_hat_train,N_lat,Nt,dt):
    
    C, A, F = getLasdiOperators(lamb,D,R)
    
    x_hat_opInf = np.zeros((11,N_lat,Nt))
    x_hat_trainFlat = np.zeros((N_lat,11*Nt))
    x_hat_opInfFlat = np.zeros((N_lat,11*Nt))
    
    for sol in range(10):
        x_hat_opInf[sol,:,0] = x_hat_train[sol,:,0]
        
        for t in range(Nt-1):
            x_hat_sq = np.zeros((int(N_lat*(N_lat+1)/2)))
            count = 0
            for j in range(N_lat):
                for k in range(j+1):
                    x_hat_sq[count] = x_hat_opInf[sol,j,t]*x_hat_opInf[sol,k,t]
                    count += 1
            
            x_hat_opInf[sol,:,t+1] = x_hat_opInf[sol,:,t] + dt*(C + A@x_hat_opInf[sol,:,t] + F@x_hat_sq)
            
        x_hat_trainFlat[:,sol*Nt:(sol+1)*Nt] = x_hat_train[sol,:,:]
        x_hat_opInfFlat[:,sol*Nt:(sol+1)*Nt] = x_hat_opInf[sol,:,:]
    
    
    error = np.linalg.norm(x_hat_opInfFlat-x_hat_trainFlat,'fro')/np.linalg.norm(x_hat_trainFlat,'fro')

    return error



N_lat = int(sys.argv[1])


N = 4148
dt = .001
Nt = 501 
nSol = 11

gamma = 1.4
tol = 1e-6

x_hist_in = np.zeros((nSol,4*N,Nt))

## make sure to fix this on the corrected models. should only use lines 385 to 391
for i in range(nSol):
    output_location = 'Cyl_x_hist_500TS_'+str(1000+100*i)+'.npy'
    volume_file = 'cell_volumes.npy'
    x_hist_temp = (np.load(output_location))
    x_hist_in[i,:N,:] = x_hist_temp[:N,:]
    x_hist_in[i,N:2*N,:] = x_hist_temp[N:2*N,:]
    x_hist_in[i,2*N:3*N,:] = x_hist_temp[2*N:3*N,:]
    x_hist_in[i,3*N:,:] = x_hist_temp[3*N:,:]
    
        
nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((4*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]

U = np.load('U_cyl.npy')
phi = U[:,:N_lat]

x_hat_train = np.zeros((nSol,N_lat,Nt))
x_hat_dot_train = np.zeros((nSol,N_lat,Nt))
D = np.zeros((1+N_lat+int((N_lat+1)*N_lat/2), (Nt)*nSol))
R = np.zeros((N_lat, (Nt)*nSol))


for sol in range(nSol):
    x_hist = np.zeros((4*N,Nt))
    x_hist[:N,:] = (x_hist_in[sol,:N,:])
    x_hist[N:2*N,:] = (x_hist_in[sol,N:2*N,:])
    x_hist[2*N:3*N,:] = (x_hist_in[sol,2*N:3*N,:])
    x_hist[3*N:,:] = (x_hist_in[sol,3*N:,:])
    
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

# lamb_all = []
lamb0_all = [-8,-7,-6,-5,-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
lamb1_all = [-8,-7,-6,-5,-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

lasdiErrors = np.zeros((len(lamb0_all),len(lamb1_all)))

for i in range(len(lamb0_all)):
    lamb[0] = lamb0_all[i]
    
    for j in range(len(lamb1_all)):
        lamb[1] = lamb1_all[j]
        
        lasdiErrors[i,j] = getLasdiError(lamb,D,R,x_hat_train,N_lat,Nt,dt)
        if np.isnan(lasdiErrors[i,j]):
            lasdiErrors[i,j]=1e8
        
i_j_min = np.unravel_index(lasdiErrors.argmin(), lasdiErrors.shape)

print('latent state dimension: ', N_lat)
print(lamb0_all[i_j_min[0]], lamb1_all[i_j_min[1]])
