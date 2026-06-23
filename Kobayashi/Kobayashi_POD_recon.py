import numpy as np

Nt = 501

tol = 1e-6

nStep = 160
x = np.linspace(0,1,nStep)
y = np.linspace(0,1,nStep)
dx = x[1] - x[0]


Nx = np.shape(x)[0]
Ny = Nx
N = Nx*Ny


x_hist_in = np.load('Kobayashi_x_hist_imp_May5_full_test.npy')
nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((2*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]
    

U, Sigma, V = np.linalg.svd(x_hist,full_matrices=False)
np.save('Kobayashi_U.npy',U[:,:20])


for i in range(5):
    N_lat = 3+i
    reconError = np.zeros(nSol)
    phi = U[:,:N_lat]
    
    
    for sol in range(nSol):
        x_hat = phi.T@x_hist_in[sol,:,:]
        x_hist_approx = phi@x_hat

        reconError[sol] = np.linalg.norm(x_hist_approx[:,:]-x_hist_in[sol,:,:],'fro')/np.linalg.norm(x_hist_in[sol,:,:],'fro')

    np.save('Kobayashi_solutions/Kobayashi_POD_recon_'+str(N_lat),reconError)

