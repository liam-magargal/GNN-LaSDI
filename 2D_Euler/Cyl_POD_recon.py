import numpy as np
import sys
import pandas as pd
import torch

def getConserved( rho, u, v, P, gamma, vol ):
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E = 1/(gamma-1)*P/rho + .5*(u*u + v*v)
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    Energy = rho * E
    
    
    return Mass, Mom_x, Mom_y, Energy, E, H

def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)

    return rho, u, v, P, E, H


def loadDataTotal(numTrain, numTest, numVal, ns, N, directory):
    x_hist = np.zeros((numTrain*ns,N,4))
    x_test = np.zeros((numTest,4*N,ns))
    gamma = 1.4
    
    for i in range(numTest):
        output_location = 'Cyl_x_hist_500TS_'+str(1050+100*i)+'.npy'
        
        volume_file = 'cell_volumes.npy'
        x_test_temp = torch.tensor(np.load(output_location))
        
        x_test[i,:,:] = x_test_temp[:,:]
        
    
    pos = None
    
    x_train = x_hist
    x_val = 0
    
    return x_train, x_val, x_test, pos


if __name__=="__main__":
    lat_dim = int(sys.argv[1])

    batchSize = 1


    N = 4148
    ns = 501
    numTrain = 11
    numTest = 10
    numVal = 0
    filename = 'output'
    directory = None
    
    x_train, x_val, x_test, pos = loadDataTotal(numTrain, numTest, numVal, ns, N, directory)
    
    gamma = 1.4

    #### svd ####
    U,Sigma,V = np.linalg.svd(x_train,full_matrices=False)
    np.save('U_cyl',U[:,:20])
    #### end svd ####
    phi = U[:,:lat_dim]
    
    rel_error = torch.zeros((ns))
    
    x_out = torch.zeros((N,4))
    
    input_hist = torch.zeros(4*N,ns)
    pred_hist = torch.zeros(4*N,ns)
    
    lat_hist = torch.zeros((ns,lat_dim))
    

    errors = np.zeros(numTest)
    for i in range(numTest):
        x_hist = x_test[i,:,:]
        x_pred = phi@phi.T@x_hist
        
        full_rel_error = np.linalg.norm(x_pred-x_hist,'fro')/np.linalg.norm(x_hist,'fro')
        errors[i] = np.linalg.norm(x_pred-x_hist,'fro')/np.linalg.norm(x_hist,'fro')
        
        print(full_rel_error)
    
    
    np.save('Cyl_solutions/Cyl_POD_recon_'+str(lat_dim),errors)
    