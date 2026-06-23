import numpy as np
import sys

import torch

from typing import Callable, Optional, Tuple

from numba import njit, jit, prange, float64, int64, float32
import numba as nb

from scipy.spatial.distance import directed_hausdorff


@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]) )
def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)

    return rho, u, v, P, E, H

@njit((float64)(float64,float64))
def np_max(val1, val2):
    if val1>val2:
        return val1
    else:
        return val2



@njit((float64[:])(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64,float64))
def getDucros(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1):
    edge_index = edge_data[:,:2]
    d0d1 = edge_data[:,2:4]
    n_hat = edge_data[:,4:6]
    faceArea = edge_data[:,6]
    edge_centers = edge_data[:,7:8]
    
    rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
    
    n_edges = np.shape(edge_index)[0]
    
    N = np.shape(cell_centroids)[0]
    
    div = np.zeros((N))
    curl = np.zeros((N))
    gradP = np.zeros((N,2))
    c = np.zeros((N))
    ducros_sensor = np.zeros((N))
    
    
    ## using RHLL flux (Nishikawa 2008)
    for i in range(n_edges):
        cell0 = int(edge_index[i,0])
        cell1 = int(edge_index[i,1])
        
        rho_in = 1.
        vx_in = u_1*np.sqrt(1.4)
        
        if cell1==-1: #
            # left boundary (inflow)
            rho_c1 = rho_in
            vx_c1 = vx_in
            vy_c1 = 0.
            P_c1 = 1.
            
        elif cell1==-2:
            rho_c1 = rho_c[cell0]
            vx_c1 = vx_c[cell0]
            vy_c1 = vy_c[cell0]
            P_c1 = P_c[cell0]
        elif cell1==-3:
            # top or bottom boundary (outflow)
            rho_c1 = rho_c[cell0]
            vx_c1 = vx_c[cell0]
            vy_c1 = vy_c[cell0]
            P_c1 = P_c[cell0]
            
        elif cell1==-4:
            # cylinder boundary (slip)
            rho_c1 = rho_c[cell0]
            k = 2.0*(vx_c[cell0]*n_hat[i,0] + vy_c[cell0]*n_hat[i,1])
            vx_c1 = (vx_c[cell0] - k*n_hat[i,0])
            vy_c1 = (vy_c[cell0] - k*n_hat[i,1])
            P_c1 = P_c[cell0]
        else:
            # normal cell
            rho_c1 = rho_c[cell1]
            vx_c1 = vx_c[cell1]
            vy_c1 = vy_c[cell1]
            P_c1 = P_c[cell1]
        
        # for the ducros sensor
        velx = vx_c1
        vely = vy_c1
        velx_cellNormal = n_hat[i,0]*velx + n_hat[i,1]*vely
        vely_cellNormal = - n_hat[i,1]*velx + n_hat[i,0]*vely
        div[cell0] = div[cell0] + velx_cellNormal * faceArea[i] / cell_volumes[cell0]
        curl[cell0] = curl[cell0] - vely_cellNormal * faceArea[i] / cell_volumes[cell0]
        
        gradP[cell0,0] = gradP[cell0,0] + P_c1*(n_hat[i,0] + n_hat[i,1]) * faceArea[i] / cell_volumes[cell0]
        gradP[cell0,1] = gradP[cell0,1] + P_c1*(-n_hat[i,1] + n_hat[i,0]) * faceArea[i] / cell_volumes[cell0] 
        
        
    for i in range(N):
        div[i] = np_max(-div[i], 0)
        c[i] = np.sqrt(gamma*P_c[i]/rho_c[i])
        
        ducros_sensor[i] = div[i] / np.sqrt(div[i]*div[i] + curl[i]*curl[i] + c[i]*c[i]) * np.sqrt( gradP[i,0]*gradP[i,0] + gradP[i,1]*gradP[i,1] ) / (P_c[i] + .01) * np.sqrt(vx_c[i]*vx_c[i] + vy_c[i]*vy_c[i])
        
    
    return ducros_sensor


N_lat = int(sys.argv[1])
testSol = int(sys.argv[2])

print('POD-LaSDI,', N_lat, testSol)

Nt = 501
dt = 1e-3
gamma = 1.4


output_location = 'Cyl_x_hist_500TS_'+str(1050+100*testSol)+'.npy'
x_hist_in = (np.load(output_location))

x_hat_hist_in = np.load('Cyl_solutions/Cyl_POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'.npy')

nSol = np.shape(x_hist_in)[0]

directory = 'Cyl_pooling_unpooling_normalized/'

pos1 = (torch.load(directory + 'pos1'))

U = np.load('U_cyl.npy')
phi = U[:,:N_lat]


mesh_file = 'Cyl_pooling_unpooling_normalized/'
edge_index = np.load(mesh_file + 'edge_index.npy') ##
cell_centroids = np.load(mesh_file + 'cell_centroids.npy')
d0d1 = np.zeros((np.shape(edge_index)[0],2))
n_hat = np.load(mesh_file + 'n_hat.npy') ##
faceArea = np.load(mesh_file + 'faceArea.npy') ##
edge_centers = np.load(mesh_file + 'edge_centers.npy') ##
edge_centers = np.zeros((np.shape(edge_index)[0],2)) 
cell_volumes = np.load(mesh_file + 'cell_volumes.npy')

N = np.shape(cell_centroids)[0]

edge_data = np.concatenate((edge_index, d0d1), axis=1)
edge_data = np.concatenate((edge_data, n_hat), axis=1)
edge_data = np.concatenate((edge_data, np.reshape(faceArea, (np.size(faceArea),1))), axis=1)
edge_data = np.concatenate((edge_data, edge_centers), axis=1)
edge_data = edge_data[edge_data[:, 0].argsort()]



x_approx = torch.zeros((Nt,N,4))
dBatSize = 501
numDbat = int(Nt/dBatSize)


x_hist_approx = phi@x_hat_hist_in
    

t_hist = np.linspace(0,.25,Nt)


interfaceError = np.zeros((Nt))
interfaceError_2norm = np.zeros((Nt))
interfaceError_sum = np.zeros((Nt))



u_1 = 1 + 0.05*testSol
pos1 = pos1.cpu().numpy()
for t in range(Nt):
    
    Mass_c = x_hist_approx[:N,t]
    Momx_c = x_hist_approx[N:2*N,t]
    Momy_c = x_hist_approx[2*N:3*N,t]
    Energy_c = x_hist_approx[3*N:,t]
    
    ducros_approx = getDucros(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1)
    
    quantile = np.quantile(a=ducros_approx,q=.95)
    ducros_approx[ducros_approx<quantile] = 0.0
    ducros_approx[ducros_approx>1e-6] = 1.
    
    
    Mass_c = x_hist_in[:N,t]
    Momx_c = x_hist_in[N:2*N,t]
    Momy_c = x_hist_in[2*N:3*N,t]
    Energy_c = x_hist_in[3*N:,t]
    ducros_GT = getDucros(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1)
    
    quantile = np.quantile(a=ducros_GT,q=.95)
    ducros_GT[ducros_GT<quantile] = 0.0
    ducros_GT[ducros_GT>1e-6] = 1.
    
    if t==250:
        np.save('Cyl_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_ducros250',ducros_approx)
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_ducros250',ducros_GT)
        np.save('Cyl_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_sol250',x_hist_approx[:,t])
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_sol250',x_hist_in[:,t])
    if t==500:
        np.save('Cyl_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_ducros500',ducros_approx)
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_ducros500',ducros_GT)
        np.save('Cyl_solutions/POD_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_sol500',x_hist_approx[:,t])
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_sol500',x_hist_in[:,t])
        
        
    pointCloud_approx = pos1[np.where(ducros_approx>0),:]
    pointCloud_GT = pos1[np.where(ducros_GT>0),:]
    
    interfaceError[t] = max(directed_hausdorff(pointCloud_approx[0,:,:], pointCloud_GT[0,:,:])[0], directed_hausdorff(pointCloud_GT[0,:,:], pointCloud_approx[0,:,:])[0])
    
    
        
np.save('Cyl_solutions/POD_LaSDI_pc_error_'+str(N_lat)+'_testSol_'+str(testSol),interfaceError)
np.save('Cyl_solutions/POD_LaSDI_pc_2norm_error_'+str(N_lat)+'_testSol_'+str(testSol),interfaceError_2norm)

