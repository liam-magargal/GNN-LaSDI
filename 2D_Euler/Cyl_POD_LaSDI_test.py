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

def getLasdiError(lamb,D,R,x_hat_train,N_lat,Nt,dt):
    
    C, A, F = getLasdiOperators(lamb,D,R)
    
    
    x_hat_opInf = np.zeros((11,N_lat,Nt))
    x_hat_trainFlat = np.zeros((N_lat,11*Nt))
    x_hat_opInfFlat = np.zeros((N_lat,11*Nt))
    
    for sol in range(11):
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
    print(error)
    
    return error


def genDomain(cell_centroids,gamma,u_1,v_1):
    N = np.shape(cell_centroids)[0]

    p1 = 1.
    rho1 = 1.
    u1 = u_1*np.sqrt(1.4)
    v1 = 0.

    rho = rho1*np.ones((N),dtype=np.float32)
    vx = u1*np.ones((N),dtype=np.float32)
    vy = v1*np.ones((N),dtype=np.float32)
    P = p1*np.ones((N),dtype=np.float32)

    return rho, vx, vy, P

def getConserved( rho, u, v, P, gamma, vol, N ):
    Energy = np.zeros((N),dtype=np.float32)
    E = np.zeros((N),dtype=np.float32)
    H = np.zeros((N),dtype=np.float32)
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E[:] = (1/(gamma-1)*P/rho + .5*(u*u + v*v))
    H[:] = ((gamma) / (gamma-1) * P / rho + .5*(u*u + v*v))
    Energy[:] = rho * E
    
    return Mass, Mom_x, Mom_y, Energy, E, H


N_lat = int(sys.argv[1])


N = 4148
dt = .001
Nt = 501
nSol = 11

gamma = 1.4

tol = 1e-6


latDimToDucros = [2, 4, 5, 6]
testSolsToDucros = [0, 3, 5, 6, 9]


x_hist_in = np.zeros((nSol,4*N,Nt))
x_hist_in_test = np.zeros((nSol-1,4*N,Nt))


for i in range(nSol):
    output_location = 'Cyl_x_hist_500TS_'+str(1000+100*i)+'.npy'
    volume_file = 'cell_volumes.npy'
    x_hist_temp = (np.load(output_location))
    x_hist_in[i,:N,:] = x_hist_temp[:N,:]
    x_hist_in[i,N:2*N,:] = x_hist_temp[N:2*N,:]
    x_hist_in[i,2*N:3*N,:] = x_hist_temp[2*N:3*N,:]
    x_hist_in[i,3*N:,:] = x_hist_temp[3*N:,:]
    
for i in range(nSol-1):
    output_location = 'Cyl_x_hist_500TS_test_'+str(1050+100*i)+'.npy'
    volume_file = 'cell_volumes.npy'
    x_hist_temp = (np.load(output_location))
    x_hist_in_test[i,:N,:] = x_hist_temp[:N,:]
    x_hist_in_test[i,N:2*N,:] = x_hist_temp[N:2*N,:]
    x_hist_in_test[i,2*N:3*N,:] = x_hist_temp[2*N:3*N,:]
    x_hist_in_test[i,3*N:,:] = x_hist_temp[3*N:,:]
    

nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((4*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]
    
U = np.load('U_cyl.npy')
phi = U[:,:N_lat]

directory = 'Cyl_pooling_unpooling_normalized/'
pos1 = (torch.load(directory + 'pos1')).numpy()

cell_volumes = np.load(directory + 'cell_volumes.npy')

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

allLamb0 = [2,-8,3,-8,-8]
allLamb1 = [7,5,6,5,5]

lamb = np.zeros(2)

lamb[0] = allLamb0[N_lat-2]
lamb[1] = allLamb1[N_lat-2]


errors = getLasdiError(lamb,D,R,x_hat_train,N_lat,Nt,dt)

C,A,F = getLasdiOperators(lamb,D,R)


x_hat_test = np.zeros((nSol,N_lat,Nt))


for sol in range(nSol-1):
    x_hist_test = x_hist_in_test[sol,:]
    x_hat_test[sol,:,:] = phi.T@x_hist_test
    
numRep = 10

errors = np.zeros(nSol-1)
lowDimSolTime = np.zeros(nSol-1)
decodeTime = np.zeros(nSol-1)
encodeTime = np.zeros(nSol-1)
totalTime = np.zeros(nSol-1)
for sol in range(nSol-1):
    for rep in range(numRep):
        ts = time.time()
        t1 = time.time()
        rho, u, v, P =  genDomain(pos1,gamma,1.05+0.1*sol,0)
        Mass, Mom_x, Mom_y, Energy, E, H = getConserved( rho, u, v, P, gamma, cell_volumes, N )
        x_init = np.zeros((4*N))
        x_init[:N] = Mass
        x_init[N:2*N] = Mom_x
        x_init[2*N:3*N] = Mom_y
        x_init[3*N:] = Energy
        x_hat_init = phi.T@x_init
        t2 = time.time()
        encodeTime[sol] += t2-t1
        
        t1 = time.time()
        x_hat_opInf = getLasdiSol(C,A,F,x_hat_init,N_lat,Nt,dt)
        t2 = time.time()
        
        x_hist_approx = phi@x_hat_opInf
        t3 = time.time()
        
        lowDimSolTime[sol] += t2-t1
        decodeTime[sol] += t3-t2

        te = time.time()
        totalTime[sol] += te-ts
        
    
    
    if sol in testSolsToDucros and N_lat in latDimToDucros:
        np.save('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat) + '_testSol_'+str(sol), x_hat_opInf)
            
    errors[sol] = np.linalg.norm(x_hist_approx[:,:]-x_hist_in_test[sol,:,:],'fro')/np.linalg.norm(x_hist_in_test[sol,:,:],'fro')
    
    
print(errors)
np.save('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat)+'_errors_',errors)
np.save('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat)+'_lowDimSolTime_',lowDimSolTime/numRep)
np.save('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat)+'_decodeTime_',decodeTime/numRep)
np.save('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat)+'_encodeTime_',encodeTime/numRep)
np.save('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat)+'_totalTime_',totalTime/numRep)

