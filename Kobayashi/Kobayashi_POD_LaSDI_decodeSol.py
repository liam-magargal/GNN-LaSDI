import numpy as np
import sys
from skimage import feature

from typing import Callable, Optional, Tuple

from scipy.spatial.distance import directed_hausdorff

N_lat = int(sys.argv[1])
testSol = int(sys.argv[2])

print('POD, ', N_lat, testSol)

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

x_hist_in = np.load('Kobayashi_x_hist_imp_May5_full_test.npy')

x_hat_hist_in = np.load('Kobayashi_solutions/Kobayashi_POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'.npy')


nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((2*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]
    

U = np.load('Kobayashi_U.npy')
phi = U[:,:N_lat]

x_hist_approx = phi@x_hat_hist_in

interfaceError = np.zeros((Nt))
interfaceError_2norm = np.zeros((Nt))
interfaceError_sum = np.zeros((Nt))

for t in range(Nt):
    cannyFilter_approx = (feature.canny(x_hist_approx[:N,t].reshape((Nx,Nx)))).reshape((Nx*Nx))
    cannyFilter_GT = (feature.canny(x_hist_in[testSol,:N,t].reshape((Nx,Nx)))).reshape((Nx*Nx))
    
    if t==250:
        np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_canny250',cannyFilter_approx.reshape((Nx,Nx)))
        np.save('Kobayashi_solutions/GT_testSol_'+str(testSol)+'_canny250',cannyFilter_GT.reshape((Nx,Nx)))
        np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_sol250',x_hist_approx[:N,t].reshape((Nx,Nx)))
        np.save('Kobayashi_solutions/GT_testSol_'+str(testSol)+'_sol250',x_hist_in[testSol,:N,t].reshape((Nx,Nx)))
    if t==500:
        np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_canny500',cannyFilter_approx.reshape((Nx,Nx)))
        np.save('Kobayashi_solutions/GT_testSol_'+str(testSol)+'_canny500',cannyFilter_GT.reshape((Nx,Nx)))
        np.save('Kobayashi_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_sol500',x_hist_approx[:N,t].reshape((Nx,Nx)))
        np.save('Kobayashi_solutions/GT_testSol_'+str(testSol)+'_sol500',x_hist_in[testSol,:N,t].reshape((Nx,Nx)))
        
        
    pointCloud_approx = coords[np.where(cannyFilter_approx>0),:]
    pointCloud_GT = coords[np.where(cannyFilter_GT>0),:]
    
    interfaceError[t] = max(directed_hausdorff(pointCloud_approx[0,:,:], pointCloud_GT[0,:,:])[0], directed_hausdorff(pointCloud_GT[0,:,:], pointCloud_approx[0,:,:])[0])

    
np.save('Kobayashi_solutions/POD_LaSDI_pc_error_'+str(N_lat)+'_testSol_'+str(testSol),interfaceError)
np.save('Kobayashi_solutions/POD_LaSDI_pc_2norm_error_'+str(N_lat)+'_testSol_'+str(testSol),interfaceError_2norm)
