import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def getSolutionImp(x_hist,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd):
    
    
    for t in range(Nt-1):
        x_curr = x_hist[:,t]
        x_next = x_hist[:,t]
        phi_curr = x_curr[:N]
        u_curr = x_curr[N:]
        u_next = x_next[N:]
        phi_next = x_next[:N]
        
        while True:
            
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
            
            
            if np.linalg.norm(r)<tol:
                x_hist[:N,t+1] = phi_next
                x_hist[N:,t+1] = u_next
                break
            else:
                J_csr = J.tocsr()
                x_next = x_next - spla.spsolve(J_csr,r)
            
    
    return x_hist


alpha = .9
gamma = 10
K = 1
tau = .0003
eps = .01
Delta = 1.

dt = .00002
Nt = 501

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


a_all = [.3875, .4625, .5375, .6125]
b_all = [.3875, .4625, .5375, .6125]


x_hist = np.zeros((16,2*N,Nt))
sol = 0
for a in a_all:
    for b in b_all:
        print(sol)
        t1 = time.time()
        r0 = .075
        r = np.sqrt(np.square(coords[:,0]-a) + np.square(coords[:,1]-b))
        phi_init = .5*(1 - np.tanh((r-r0)/(2*eps*np.sqrt(2))))

        U_init = Delta*phi_init

        x_hist_sol = np.zeros((2*N,Nt))
        x_hist_sol[:N,0] = phi_init
        x_hist_sol[N:,0] = U_init
        x_hist_sol = getSolutionImp(x_hist_sol,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd)
        x_hist[sol,:,:] = x_hist_sol.copy()
        
        t2 = time.time()
        print('Solution time: ', t2-t1)
        

np.save('Kobayashi_x_hist_imp_May5_full_test.npy',x_hist)
