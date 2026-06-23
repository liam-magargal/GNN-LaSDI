import numpy as np
import sys
# import matplotlib.pyplot as plt
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy
# from skimage import feature
# import cupyx
# import cupy
# from cupy import cupyx
# from cupyx import scipy
# from cupy import cupyx
# from cupyx import scipy
# 
# import cupyx.scipy.sparse.linalg as cuspla
# import torch
# import torch.sparse as tsp
# import torch.sparse.spsolve as tspsolve


def getSolutionImpExp(x_hist,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd):
    
    # r = np.zeros((N))
    # J = sp.dok_matrix((N, N))
    
    # x_next = np.zeros(2*N)
    
    for t in range(Nt-1):
        # print(t)
        x_curr = x_hist[:,t]
        phi_curr = x_curr[:N]
        u_curr = x_curr[N:]
        u_next = x_curr[N:]
        phi_next = np.zeros((N))
        # print(u_curr[intInd])
        
        m = alpha/np.pi*np.arctan(gamma*(Delta - u_curr))
        # m = 1/2*np.ones(N)
        phi_next[intInd] = phi_curr[intInd] + dt/tau*(eps*eps/dx/dx*(phi_curr[intInd-1] - 2*phi_curr[intInd] + phi_curr[intInd+1]) + eps*eps/dx/dx*(phi_curr[intInd-Nx] - 2*phi_curr[intInd] + phi_curr[intInd+Nx]) - 1/2*phi_curr[intInd] + 3/2*phi_curr[intInd]**2 - phi_curr[intInd]**3 + m[intInd]*(phi_curr[intInd] - phi_curr[intInd]**2))
        # phi_next[intInd] = phi_curr[intInd] + dt/tau*(eps*eps/dx/dx*(phi_curr[intInd-1] - 2*phi_curr[intInd] + phi_curr[intInd+1]) - 1/2*phi_curr[intInd] + 3/2*phi_curr[intInd]**2 - phi_curr[intInd]**3 + m[intInd]*(phi_curr[intInd] - phi_curr[intInd]**2))
        # + a*(np.random.rand(N-2) - 1/2)*phi_curr[intInd]*(1-phi_curr[intInd])
        # print(a*(np.random.rand(N-2) - 1/2)*phi_curr[intInd]*(1-phi_curr[intInd]))
        
        # update boundary points
        # phi_next[0] = 4/3*phi_next[1] - 1/3*phi_next[2]
        # phi_next[-1] = 4/3*phi_next[-2] - 1/3*phi_next[-3]
        phi_next[boundInd] = phi_curr[boundInd]
        # phi_next[0] = phi_curr[0]
        # phi_next[-1] = phi_curr[-1]
        
        
        # m = 1/2*np.ones(len(intInd))
        while True:
            r = np.zeros((N))
            J = sp.dok_matrix((N, N))
            
            t1 = time.time()
            
            m = alpha/np.pi*np.arctan(gamma*(Delta - u_next))
            
            # for u
            r[intInd] = u_next[intInd] - u_curr[intInd] - dt*(1/dx/dx*(u_next[intInd-1] - 2*u_next[intInd] + u_next[intInd+1]) + 1/dx/dx*(u_next[intInd-Nx] - 2*u_next[intInd] + u_next[intInd+Nx]) + K/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2)))
            J[intInd,intInd] = 1 - dt*(1/dx/dx*(-4)) - dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))
            J[intInd,intInd-1] = - dt*(1/dx/dx)
            J[intInd,intInd+1] = - dt*(1/dx/dx)
            J[intInd,intInd-Nx] = - dt*(1/dx/dx)
            J[intInd,intInd+Nx] = - dt*(1/dx/dx)
            
            # r[boundInd] = 1 - u_next[boundInd]
            # J[boundInd,boundInd] = -1
            
            r[boundInd] = u_next[boundInd]
            J[boundInd,boundInd] = 1
            
            
            
            t2 = time.time()
            
            # print(np.linalg.norm(r), np.linalg.norm(r[:N]), np.linalg.norm(r[N:]), tol)
            # sys.exit()
            
            # t1 = time.time()
            if np.linalg.norm(r)<tol:
                # print(phi_next-phi_curr)
                x_hist[:N,t+1] = phi_next
                x_hist[N:,t+1] = u_next
                break
            else:
                J_csr = J.tocsr()
                u_next = u_next - spla.spsolve(J_csr,r)
            
            # t2 = time.time()
    
    return x_hist


def getSolutionImp(x_hist,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd):
    
    
    for t in range(Nt-1):
        # print(t)
        x_curr = x_hist[:,t]
        x_next = x_hist[:,t]
        phi_curr = x_curr[:N]
        u_curr = x_curr[N:]
        u_next = x_next[N:]
        phi_next = x_next[:N]
        
        while True:
            # t1 = time.time()
            
            r = np.zeros((2*N))
            J = sp.dok_matrix((2*N, 2*N))
            
            u_next = x_next[N:]
            phi_next = x_next[:N]
            
            m = alpha/np.pi*np.arctan(gamma*(Delta - u_next))
            
            r[intInd] = phi_next[intInd] - phi_curr[intInd] - dt/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2))
            r[boundInd] = phi_next[boundInd] 
            
            J[intInd,intInd] = 1 - dt/tau*(-4*eps*eps/dx/dx - 1/2 + 3*phi_next[intInd] -3*phi_next[intInd]**2 + m[intInd]*(1 - 2*phi_next[intInd]))
            J[intInd,intInd-1] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,intInd+1] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,intInd-Nx] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,intInd+Nx] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,N+intInd] = - dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))
            J[boundInd,boundInd] = 1
            
            
            r[N+intInd] = u_next[intInd] - u_curr[intInd] - dt*(1/dx/dx*(u_next[intInd-1] - 2*u_next[intInd] + u_next[intInd+1]) + 1/dx/dx*(u_next[intInd-Nx] - 2*u_next[intInd] + u_next[intInd+Nx]) + K/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2)))
            J[N+intInd,N+intInd] = 1 - dt*(1/dx/dx*(-4)) - dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))
            J[N+intInd,N+intInd-1] = - dt*(1/dx/dx)
            J[N+intInd,N+intInd+1] = - dt*(1/dx/dx)
            J[N+intInd,N+intInd-Nx] = - dt*(1/dx/dx)
            J[N+intInd,N+intInd+Nx] = - dt*(1/dx/dx)
            
            J[N+intInd,intInd] = -K*dt/tau*(-4*eps*eps/dx/dx - 1/2 + 3*phi_next[intInd] -3*phi_next[intInd]**2 + m[intInd]*(1 - 2*phi_next[intInd]))
            J[N+intInd,intInd-1] = -K*dt/tau*(eps*eps/dx/dx)
            J[N+intInd,intInd+1] = -K*dt/tau*(eps*eps/dx/dx)
            J[N+intInd,intInd-Nx] = -K*dt/tau*(eps*eps/dx/dx)
            J[N+intInd,intInd+Nx] = -K*dt/tau*(eps*eps/dx/dx)
            
            r[N+boundInd] = u_next[boundInd]
            J[N+boundInd,N+boundInd] = 1
            
            
            # t2 = time.time()
            
            # print('time to set up: ', t2-t1)
            # print(np.linalg.norm(r), np.linalg.norm(r[:N]), np.linalg.norm(r[N:]), tol)
            # print(np.linalg.norm(r), tol)
            # sys.exit()
            
            # t1 = time.time()
            if np.linalg.norm(r)<tol:
                # print(phi_next-phi_curr)
                x_hist[:N,t+1] = phi_next
                x_hist[N:,t+1] = u_next
                break
            else:
                J_csr = J.tocsr()
                x_next = x_next - spla.spsolve(J_csr,r)
            
            # t2 = time.time()
            # print('time to solve: ', t2-t1)
    
    return x_hist


def getSolutionImpAlternativeCupy(x_hist,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd):
    
    cupy.array([1])
    for t in range(Nt-1):
        print(t)
        x_curr = x_hist[:,t]
        x_next = x_hist[:,t]
        phi_curr = x_curr[:N]
        u_curr = x_curr[N:]
        u_next = x_next[N:]
        phi_next = x_next[:N]
        
        while True:
            
            t1 = time.time()
            # r = np.zeros((2*N))
            r = cupy.zeros((2*N))
            # J = sp.dok_matrix((2*N, 2*N))
            # J = sp.dok_matrix((2*N, 2*N))
            
            leftInd = []
            rightInd = []
            values = []
            
            
            u_next = x_next[N:]
            phi_next = x_next[:N]
            
            m = alpha/np.pi*np.arctan(gamma*(Delta - u_next))
            
            r[intInd] = phi_next[intInd] - phi_curr[intInd] - dt/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2))
            r[boundInd] = phi_next[boundInd] 
            
            
            leftInd = intInd.tolist() + intInd.tolist() + intInd.tolist() + intInd.tolist() + intInd.tolist() + intInd.tolist() + boundInd.tolist()
            rightInd = intInd.tolist() + (intInd-1).tolist() + (intInd+1).tolist() + (intInd-Nx).tolist() + (intInd+Nx).tolist() + (intInd+N).tolist() + boundInd.tolist()
            
            values = values + (1 - dt/tau*(-4*eps*eps/dx/dx - 1/2 + 3*phi_next[intInd] -3*phi_next[intInd]**2 + m[intInd]*(1 - 2*phi_next[intInd]))).tolist()
            values = values + [-dt/tau*(eps*eps/dx/dx)]*len(intInd)*4
            values = values + (- dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))).tolist()
            values = values + [1]*len(boundInd)
            
            
            r[N+intInd] = u_next[intInd] - u_curr[intInd] - dt*(1/dx/dx*(u_next[intInd-1] - 2*u_next[intInd] + u_next[intInd+1]) + 1/dx/dx*(u_next[intInd-Nx] - 2*u_next[intInd] + u_next[intInd+Nx]) + K/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2)))
            
            leftInd = leftInd + (N+intInd).tolist() + (N+intInd).tolist() + (N+intInd).tolist() + (N+intInd).tolist() + (N+intInd).tolist()
            rightInd = rightInd + (N+intInd).tolist() + (N+intInd-1).tolist() + (N+intInd+1).tolist() + (N+intInd-Nx).tolist() + (N+intInd+Nx).tolist()
            values = values + (1 - dt*(1/dx/dx*(-4)) - dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))).tolist()
            values = values + [-dt*(1/dx/dx)]*len(intInd)*4
            
            
            leftInd = leftInd + (N+intInd).tolist() + (N+intInd).tolist() + (N+intInd).tolist() + (N+intInd).tolist() + (N+intInd).tolist()
            rightInd = rightInd + (intInd).tolist() + (intInd-1).tolist() + (intInd+1).tolist() + (intInd-Nx).tolist() + (intInd+Nx).tolist()
            values = values + (-K*dt/tau*(-4*eps*eps/dx/dx - 1/2 + 3*phi_next[intInd] -3*phi_next[intInd]**2 + m[intInd]*(1 - 2*phi_next[intInd]))).tolist()
            values = values + [-K*dt*(eps*eps/dx/dx)]*len(intInd)*4
            
            
            
            leftInd = leftInd + (N+boundInd).tolist()
            rightInd = rightInd + (N+boundInd).tolist()
            values = values + [1]*len(boundInd)
            
            # J = sp.coo_matrix((np.array(values),(np.array(leftInd),np.array(rightInd))))
            # J[np.array(leftInd),np.array(rightInd)] = np.array(values)
            
            t1 = time.time()
            J = cupyx.scipy.sparse.coo_matrix((cupy.array(values),(cupy.array(leftInd),cupy.array(rightInd))),shape=(2*N,2*N))
            # J = scipy.sparse.coo_matrix((np.array(values),(np.array(leftInd),np.array(rightInd))),shape=(2*N,2*N))
            
            
            t2 = time.time()
            print('time to set up: ', t2-t1)
            
            # print(np.linalg.norm(r), np.linalg.norm(r[:N]), np.linalg.norm(r[N:]), tol)
            print(np.linalg.norm(r), tol)
            # sys.exit()
            
            t1 = time.time()
            if np.linalg.norm(r)<tol:
                # print(phi_next-phi_curr)
                x_hist[:N,t+1] = phi_next
                x_hist[N:,t+1] = u_next
                break
            else:
                J_csr = J.tocsr()
                x_next = x_next - (cuspla.spsolve(J_csr,r)).get()
                # p = (cuspla.spsolve(J_csr,r))
                # x_next = x_next - spla.spsolve(J_csr,r)
            
            t2 = time.time()
            print('time to solve: ', t2-t1)
    
    return x_hist




rep = int(sys.argv[1])

# m = 1/2 #find out how m is dependent on T and Te
alpha = .9
gamma = 10
# a = 0.01
K = 1
tau = .0003
eps = .01
Delta = 1.

dt = .00002 #.0002
Nt = 501 # 500

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



# a = .1
# b = .1

#a_all = [.0375, .0625, .0875, .1125]
#b_all = [.0375, .0625, .0875, .1125]

# a_all = [.425, .475, .525, .575]
# b_all = [.425, .475, .525, .575]

#a_all = [.35, .45, .55, .65]
#b_all = [.35, .45, .55, .65]

a_all = [.3875, .4625, .5375, .6125]
b_all = [.3875, .4625, .5375, .6125]


# a_all = [.025]
# b_all = [.025]

initTime = np.zeros(16)
solTime = np.zeros(16)

x_hist = np.zeros((16,2*N,Nt))
sol = 0
for a in a_all:
    for b in b_all:
        print(sol)
        t1 = time.time()
        r0 = .075
        r = np.sqrt(np.square(coords[:,0]-a) + np.square(coords[:,1]-b))
        phi_init = .5*(1 - np.tanh((r-r0)/(2*eps*np.sqrt(2))))

#        r = np.square(coords[:,0]-0.5)/a/a + np.square(coords[:,1]-0.5)/b/b
 #       phi_init = .5*(1 - np.tanh(2.5*(r-1)))
        U_init = Delta*phi_init
        t2 = time.time()

#        initialStateMask = np.where(np.square(coords[:,0]-0.5)/a/a+np.square(coords[:,1]-0.5)/b/b<1)[0]
 #       phi_init = np.zeros((N))
  #      phi_init[initialStateMask] = np.ones((len(initialStateMask)))
   #     U_init = Delta*phi_init
        x_hist_sol = np.zeros((2*N,Nt))
        x_hist_sol[:N,0] = phi_init
        x_hist_sol[N:,0] = U_init
        # x_hist_sol = getSolutionImpExp(x_hist_sol,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd)
        x_hist_sol = getSolutionImp(x_hist_sol,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd)
        # x_hist_sol = getSolutionImpAlternativeCupy(x_hist_sol,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd)
        # x_hist_sol = getSolutionImpAlternativeTorch(x_hist_sol,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd)
        x_hist[sol,:,:] = x_hist_sol.copy()
        
        t3 = time.time()
        
        initTime[sol] = t2-t1
        solTime[sol] = t3-t2

        print('Solution time: ', t3-t1)
        
        # plt.figure(1)
        # plt.imshow(x_hist_sol[:N,0].reshape((Nx,Nx)))
        # plt.figure(2)
        # plt.imshow(x_hist_sol[:N,999].reshape((Nx,Nx)))
        # plt.show()
        
        # sys.exit()
        sol += 1

np.save('AllenCahn_FOM_initTime_'+str(rep),initTime)
np.save('AllenCahn_FOM_solTime_'+str(rep),solTime)

# np.save('AllenCahn_x_hist_imp_May5_full_test.npy',x_hist)
# np.save('AllenCahn_x_hist_imp_March17_singleLong.npy',x_hist)
