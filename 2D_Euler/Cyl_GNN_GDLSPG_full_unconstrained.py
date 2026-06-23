#!/usr/bin/env python3
import numpy as np
import sys
import time
import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


from numba import njit, jit, prange, float64, int64
import numba as nb

def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, assign_index: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.
    For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
    interpolated features :math:`\mathbf{f}(y)` are given by

    .. math::
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}

    and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
    to :math:`y`.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        pos_x (torch.Tensor): Node position matrix
            :math:`\in \mathbb{R}^{N \times d}`.
        pos_y (torch.Tensor): Upsampled node position matrix
            :math:`\in \mathbb{R}^{M \times d}`.
        batch_x (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{X}` to a specific example.
            (default: :obj:`None`)
        batch_y (torch.Tensor, optional): Batch vector
            :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{Y}` to a specific example.
            (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    """

    with torch.no_grad():
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter(x[:,x_idx] * weights, y_idx, 1, pos_y.size(0), reduce='sum')
    y = y / scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')

    return y


class architecture(nn.Module):
    def __init__(self,N_lat):
        super().__init__()
        
        self.conv1_1 = torch_geometric.nn.SAGEConv(4,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv1_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv1_1.lin_r.weight)
        
        self.conv1_2 = torch_geometric.nn.SAGEConv(16,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv1_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv1_2.lin_r.weight)
        
        self.conv2_1 = torch_geometric.nn.SAGEConv(16,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv2_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv2_1.lin_r.weight)
        
        self.conv2_2 = torch_geometric.nn.SAGEConv(64,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv2_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv2_2.lin_r.weight)
        
        self.conv3_1 = torch_geometric.nn.SAGEConv(64,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv3_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv3_1.lin_r.weight)
        
        self.conv3_2 = torch_geometric.nn.SAGEConv(128,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv3_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv3_2.lin_r.weight)
        
        self.conv4_1 = torch_geometric.nn.SAGEConv(128,256,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv4_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv4_1.lin_r.weight)
        
        self.conv4_2 = torch_geometric.nn.SAGEConv(256,256,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv4_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv4_2.lin_r.weight)
        
        self.layer5 = torch.nn.Linear(256*2,N_lat,bias=False)
        nn.init.xavier_uniform_(self.layer5.weight.data)
        
        self.layer6 = torch.nn.Linear(N_lat,2*256,bias=False)
        nn.init.xavier_uniform_(self.layer6.weight.data)
        
        self.conv7_1 = torch_geometric.nn.SAGEConv(256,256,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv7_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv7_1.lin_r.weight)
        
        self.conv7_2 = torch_geometric.nn.SAGEConv(256,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv7_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv7_2.lin_r.weight)
        
        self.conv8_1 = torch_geometric.nn.SAGEConv(128,128,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv8_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv8_1.lin_r.weight)
        
        self.conv8_2 = torch_geometric.nn.SAGEConv(128,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv8_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv8_2.lin_r.weight)
        
        self.conv9_1 = torch_geometric.nn.SAGEConv(64,64,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv9_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv9_1.lin_r.weight)
        
        self.conv9_2 = torch_geometric.nn.SAGEConv(64,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv9_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv9_2.lin_r.weight)
        
        self.conv10_1 = torch_geometric.nn.SAGEConv(16,16,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_1.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_1.lin_r.weight)
        
        self.conv10_2 = torch_geometric.nn.SAGEConv(16,4,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_2.lin_r.weight)
        
        
    def encode(self,x,batchSize,e1,e2,e3,e4,e5,s1,s2,s3,s4):
        
        x[:,:,0] = (x[:,:,0] - .5) / 3.
        x[:,:,1] = (x[:,:,1]) / 4.
        x[:,:,2] = (x[:,:,2] + 3.) / 6.
        x[:,:,3] = (x[:,:,3] - 2.) / 17.

        x = torch.nn.functional.elu(self.conv1_1(x,e1))
        x = torch.nn.functional.elu(self.conv1_2(x,e1))
        x = torch.matmul(s1.transpose(1, 2), x)
        
        x = torch.nn.functional.elu(self.conv2_1(x,e2))
        x = torch.nn.functional.elu(self.conv2_2(x,e2))
        x = torch.matmul(s2.transpose(1, 2), x)
        
        x = torch.nn.functional.elu(self.conv3_1(x,e3))
        x = torch.nn.functional.elu(self.conv3_2(x,e3))
        x = torch.matmul(s3.transpose(1, 2), x)
        
        x = torch.nn.functional.elu(self.conv4_1(x,e4))
        x = torch.nn.functional.elu(self.conv4_2(x,e4))
        x = torch.matmul(s4.transpose(1, 2), x)
        
        x = x.reshape((batchSize,256*2))
        x = self.layer5(x)
        return x
    
    
    def decode(self,x,batchSize,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5, ai_54, ai_43, ai_32, ai_21):
        
        x = torch.nn.functional.elu(self.layer6(x))
        
        x = x.reshape((batchSize,2,256))
        
        x = knn_interpolate(x,pos5,pos4,assign_index=ai_54)
    
        x = torch.nn.functional.elu(self.conv7_1(x,e4))
        x = torch.nn.functional.elu(self.conv7_2(x,e4))
        x = knn_interpolate(x,pos4,pos3,assign_index=ai_43)
        
        x = torch.nn.functional.elu(self.conv8_1(x,e3))
        x = torch.nn.functional.elu(self.conv8_2(x,e3))
        
        x = knn_interpolate(x,pos3,pos2,assign_index=ai_32)
        
        x = torch.nn.functional.elu(self.conv9_1(x,e2))
        x = torch.nn.functional.elu(self.conv9_2(x,e2))
        
        x = knn_interpolate(x,pos2,pos1,assign_index=ai_21)
        
        x = torch.nn.functional.elu(self.conv10_1(x,e1))
        x = self.conv10_2(x,e1)
        
        x[:,:,0] = (x[:,:,0] * 3.) + .5
        x[:,:,1] = (x[:,:,1]) * 4.
        x[:,:,2] = (x[:,:,2] * 6.) - 3.
        x[:,:,3] = (x[:,:,3] * 17) + 2.

        x = x.reshape((batchSize,4148,4))
        
        return x
    
    



def genDomain(cell_centroids,gamma,u_1,v_1):
    N = np.shape(cell_centroids)[0]

    p1 = 1.
    rho1 = 1.
    u1 = u_1*np.sqrt(1.4)
    v1 = 0.

    rho = rho1*np.ones((N),dtype=np.float64)
    vx = u1*np.ones((N),dtype=np.float64)
    vy = v1*np.ones((N),dtype=np.float64)
    P = p1*np.ones((N),dtype=np.float64)

    return rho, vx, vy, P



@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:],int64) )
def fromState(x_curr, N):
    Mass = x_curr[:N]
    Momx = x_curr[N:2*N]
    Momy = x_curr[2*N:3*N]
    Energy = x_curr[3*N:]
    
    return Mass, Momx, Momy, Energy

@jit((float64[:])(float64[:],float64[:],float64[:],float64[:],int64) )
def toState(Mass, Momx, Momy, Energy, N):
    x = np.zeros((4*N),dtype=np.float64)
    x[:N] = Mass
    x[N:2*N] = Momx
    x[2*N:3*N] = Momy
    x[3*N:] = Energy
    
    return x

@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]) )
def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    
    return rho, u, v, P, E, H

@jit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))(float64[:],float64[:],float64[:],float64[:],float64,float64[:]) )
def getConserved( rho, u, v, P, gamma, vol ):
    Mass   = rho
    Mom_x  = rho * u
    Mom_y  = rho * v
    E = (1/(gamma-1)*P/rho + .5*(u*u + v*v))
    H = ((gamma) / (gamma-1) * P / rho + .5*(u*u + v*v))
    Energy = rho * E
    
    return Mass, Mom_x, Mom_y, Energy, E, H

@njit((float64)(float64,float64))
def np_max(val1, val2):
    if val1>val2:
        return val1
    else:
        return val2

@njit((float64)(float64,float64))
def np_min(val1, val2):
    if val2>val1:
        return val1
    else:
        return val2


@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:],int64))
def getResidual(x_next, Mass_n, Momx_n, Momy_n, Energy_n,N):
    f = np.zeros((4*N))
    f[:N] = x_next[:N] - Mass_n
    f[N:2*N] = x_next[N:2*N] - Momx_n
    f[2*N:3*N] = x_next[2*N:3*N] - Momy_n
    f[3*N:] = x_next[3*N:] - Energy_n
    
    return f
    
@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64,float64))
def getVelocity(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1):
    edge_index = edge_data[:,:2]
    d0d1 = edge_data[:,2:4]
    n_hat = edge_data[:,4:6]
    faceArea = edge_data[:,6]
    edge_centers = edge_data[:,7:8]

    rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)

    n_edges = np.shape(edge_index)[0]

    N = np.shape(cell_centroids)[0]

    vel_Mass = np.zeros((N))
    vel_Momx = np.zeros((N))
    vel_Momy = np.zeros((N))
    vel_Energy = np.zeros((N))

    ## using RHLL flux (Nishikawa 2008)
    for i in range(n_edges):
        cell0 = int(edge_index[i,0])
        cell1 = int(edge_index[i,1])

        rho_in = 1.
        vx_in = u_1*np.sqrt(1.4) 
        
        if cell1==-1:
            # left boundary (inflow)
            rho_c1 = rho_in
            vx_c1 = vx_in
            vy_c1 = 0.
            P_c1 = 1.

        elif cell1==-2:
            # right boundary (outflow)
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


        E_c1 = 1/(gamma-1)*P_c1/rho_c1 + .5*(vx_c1*vx_c1 + vy_c1*vy_c1)
        H_c1 = (gamma) / (gamma-1) * P_c1 / rho_c1 + .5*(vx_c1*vx_c1 + vy_c1*vy_c1)


        # Get direction for Rotated Roe flux (from Ren 2003)
        n1 = np.zeros((2))
        n2 = np.zeros((2))
        n1_tilde = np.zeros((2))
        n2_tilde = np.zeros((2))
        
        epsilon = 1e-12*1.
        if np.sqrt((vx_c[cell0]-vx_c1)*(vx_c[cell0]-vx_c1) + (vy_c[cell0]-vy_c1)*(vy_c[cell0]-vy_c1)) <= epsilon:
            n1[0] = n_hat[i,0]
            n1[1] = n_hat[i,1]
        else:
            n1[0] = (vx_c1-vx_c[cell0]) / np.sqrt((vx_c1-vx_c[cell0])*(vx_c1-vx_c[cell0]) + (vy_c1-vy_c[cell0])*(vy_c1-vy_c[cell0]))
            n1[1] = (vy_c1-vy_c[cell0]) / np.sqrt((vx_c1-vx_c[cell0])*(vx_c1-vx_c[cell0]) + (vy_c1-vy_c[cell0])*(vy_c1-vy_c[cell0]))


        n2[0] = -n1[1]
        n2[1] = n1[0]


        alpha1 = n_hat[i,0] * n1[0] + n_hat[i,1] * n1[1]
        alpha2 = n_hat[i,0] * n2[0] + n_hat[i,1] * n2[1]


        if alpha1 < 0:
            n1_tilde[0] = -1*n1[0]
            n1_tilde[1] = -1*n1[1]
        else:
            n1_tilde[0] = n1[0]
            n1_tilde[1] = n1[1]
        if alpha2 < 0:
            n2_tilde[0] = -1*n2[0]
            n2_tilde[1] = -1*n2[1]
        else:
            n2_tilde[0] = n2[0]
            n2_tilde[1] = n2[1]

        alpha1 = n_hat[i,0] * n1_tilde[0] + n_hat[i,1] * n1_tilde[1]
        alpha2 = n_hat[i,0] * n2_tilde[0] + n_hat[i,1] * n2_tilde[1]


        # align cell state values with their normal vectors
        # hold onto these cartesian values to reassign in the future
        vx_cart0 = (vx_c[cell0])
        vy_cart0 = (vy_c[cell0])
        vx_cart1 = vx_c1
        vy_cart1 = vy_c1

        # # rotate velocity components to align with cell face
        vx_c[cell0] = n_hat[i,0]*vx_cart0 + n_hat[i,1]*vy_cart0
        vy_c[cell0] = - n_hat[i,1]*vx_cart0 + n_hat[i,0]*vy_cart0
        vx_c1 = n_hat[i,0]*vx_cart1 + n_hat[i,1]*vy_cart1
        vy_c1 = - n_hat[i,1]*vx_cart1 + n_hat[i,0]*vy_cart1


        # get nominal cell-centered fluxes based on rotated states
        # compute fluxes (based on cell 0)
        flux_Mass_0 = rho_c[cell0]*vx_c[cell0]
        flux_Momx_0 = rho_c[cell0]*vx_c[cell0]*vx_c[cell0] + P_c[cell0]
        flux_Momy_0 = rho_c[cell0]*vx_c[cell0]*vy_c[cell0]
        flux_Energy_0 = rho_c[cell0]*vx_c[cell0]*H_c[cell0]


        # compute fluxes (based on cell 1)
        flux_Mass_1 = rho_c1*vx_c1
        flux_Momx_1 = rho_c1*vx_c1*vx_c1 + P_c1
        flux_Momy_1 = rho_c1*vx_c1*vy_c1
        flux_Energy_1 = rho_c1*vx_c1*H_c1


        # rotate fluxes back
        flux_Momx_0_r = flux_Momx_0
        flux_Momy_0_r = flux_Momy_0
        flux_Momx_0 = flux_Momx_0_r * n_hat[i,0] - flux_Momy_0_r * n_hat[i,1]
        flux_Momy_0 = flux_Momx_0_r * n_hat[i,1] + flux_Momy_0_r * n_hat[i,0]

        flux_Momx_1_r = flux_Momx_1
        flux_Momy_1_r = flux_Momy_1
        flux_Momx_1 = flux_Momx_1_r * n_hat[i,0] - flux_Momy_1_r * n_hat[i,1]
        flux_Momy_1 = flux_Momx_1_r * n_hat[i,1] + flux_Momy_1_r * n_hat[i,0]


        # rotate velocities back to cartesian values
        vx_c[cell0] = vx_cart0
        vy_c[cell0] = vy_cart0
        vx_c1 = vx_cart1
        vy_c1 = vy_cart1
        # print('returned: ', vy_c[cell0])


        # Get Roe-averaged state at boundary
        rho_hat = np.sqrt(rho_c[cell0]*rho_c1)
        u_hat = (vx_c[cell0]*np.sqrt(rho_c[cell0]) + vx_c1*np.sqrt(rho_c1)) / (np.sqrt(rho_c[cell0]) + np.sqrt(rho_c1))
        v_hat = (vy_c[cell0]*np.sqrt(rho_c[cell0]) + vy_c1*np.sqrt(rho_c1)) / (np.sqrt(rho_c[cell0]) + np.sqrt(rho_c1))
        H_hat = (H_c[cell0]*np.sqrt(rho_c[cell0]) + H_c1*np.sqrt(rho_c1)) / (np.sqrt(rho_c[cell0]) + np.sqrt(rho_c1))
        c_hat = np.sqrt((gamma-1)*(H_hat - (u_hat*u_hat + v_hat*v_hat)/2))

        # get SRp and SLm
        qn_L = vx_c[cell0] * n1_tilde[0] + vy_c[cell0] * n1_tilde[1]
        qn_R = vx_c1 * n1_tilde[0] + vy_c1 * n1_tilde[1]
        c_L = np.sqrt((gamma-1)*(H_c[cell0] - (vx_c[cell0]*vx_c[cell0] + vy_c[cell0]*vy_c[cell0])/2))
        c_R = np.sqrt((gamma-1)*(H_c1 - (vx_c1*vx_c1 + vy_c1*vy_c1)/2))
        qn_hat_n1 = u_hat * n1_tilde[0] + v_hat * n1_tilde[1]


        # get SL and SR
        SL = np_min(qn_L-c_L, qn_hat_n1-c_hat)
        SR = np_max(qn_R+c_R, qn_hat_n1+c_hat)

        SRp = np_max(0,SR)
        SLm = np_min(0,SL)

        flux_Mass = (SRp*flux_Mass_0 - SLm*flux_Mass_1) / (SRp - SLm)
        flux_Momx = (SRp*flux_Momx_0 - SLm*flux_Momx_1) / (SRp - SLm)
        flux_Momy = (SRp*flux_Momy_0 - SLm*flux_Momy_1) / (SRp - SLm)
        flux_Energy = (SRp*flux_Energy_0 - SLm*flux_Energy_1) / (SRp - SLm)
        
        
        # get eigenvalues (based on n2_tilde)
        qn_hat_n2 = u_hat * n2_tilde[0] + v_hat * n2_tilde[1]
        lambda1_n2 = qn_hat_n2 - c_hat
        lambda2_n2 = qn_hat_n2
        lambda3_n2 = qn_hat_n2 + c_hat
        lambda4_n2 = qn_hat_n2

        # get lambda_star terms (for all k, not just k=1,3)
        delta = .2
        if np.abs(lambda1_n2) >= delta:
            lambda1_n2_star = np.abs(lambda1_n2)
        else:
            lambda1_n2_star = 1/(2*delta) * (np.abs(lambda1_n2)*np.abs(lambda1_n2) + delta*delta)

        if np.abs(lambda2_n2) >= delta:
            lambda2_n2_star = np.abs(lambda2_n2)
        else:
            lambda2_n2_star = 1/(2*delta) * (np.abs(lambda2_n2)*np.abs(lambda2_n2) + delta*delta)


        if np.abs(lambda3_n2) >= delta:
            lambda3_n2_star = np.abs(lambda3_n2)
        else:
            lambda3_n2_star = 1/(2*delta) * (np.abs(lambda3_n2)*np.abs(lambda3_n2) + delta*delta)

        if np.abs(lambda4_n2) >= delta:
            lambda4_n2_star = np.abs(lambda4_n2)
        else:
            lambda4_n2_star = 1/(2*delta) * (np.abs(lambda4_n2)*np.abs(lambda4_n2) + delta*delta)


        s_hat_1_RHLL = alpha2*lambda1_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda1_n2 + 2*alpha1*SRp*SLm)
        s_hat_2_RHLL = alpha2*lambda2_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda2_n2 + 2*alpha1*SRp*SLm)
        s_hat_3_RHLL = alpha2*lambda3_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda3_n2 + 2*alpha1*SRp*SLm)
        s_hat_4_RHLL = alpha2*lambda4_n2_star - (1/(SRp - SLm)) * (alpha2*(SRp+SLm)*lambda4_n2 + 2*alpha1*SRp*SLm)


        drho = rho_c1 - rho_c[cell0]
        dP = P_c1 - P_c[cell0]

        # the diffusion term here is computed wrt n2_tilde, as indicated by the RHLL flux in the paper
        dq_n = (vx_c1 - vx_c[cell0]) * n2_tilde[0] + (vy_c1 - vy_c[cell0]) * n2_tilde[1]
        dq_t = (vx_c1 - vx_c[cell0]) * (-1*n2_tilde[1]) + (vy_c1 - vy_c[cell0]) * n2_tilde[0]


        w1_n2 = (dP - rho_hat*c_hat*dq_n)/(2*c_hat*c_hat)
        w2_n2 = drho - dP/(c_hat*c_hat)
        w3_n2 = (dP + rho_hat*c_hat*dq_n)/(2*c_hat*c_hat)
        w4_n2 = rho_hat*dq_t

        qn_hat = u_hat*n2_tilde[0] + v_hat*n2_tilde[1]
        qt_hat = u_hat*(-1*n2_tilde[1]) + v_hat*n2_tilde[0]

        flux_Mass = flux_Mass - 1/2 * (s_hat_1_RHLL * w1_n2 * 1 + s_hat_2_RHLL * w2_n2 * 1 + s_hat_3_RHLL * w3_n2 * 1)
        flux_Momx = flux_Momx - 1/2 * (s_hat_1_RHLL * w1_n2 * (u_hat-c_hat*n2_tilde[0]) + s_hat_2_RHLL * w2_n2 * (u_hat) + s_hat_3_RHLL * w3_n2 * (u_hat+c_hat*n2_tilde[0]) + s_hat_4_RHLL * w4_n2 * (-n2_tilde[1]))
        flux_Momy = flux_Momy - 1/2 * (s_hat_1_RHLL * w1_n2 * (v_hat-c_hat*n2_tilde[1]) + s_hat_2_RHLL * w2_n2 * (v_hat) + s_hat_3_RHLL * w3_n2 * (v_hat+c_hat*n2_tilde[1]) + s_hat_4_RHLL * w4_n2 * (n2_tilde[0]))
        flux_Energy = flux_Energy - 1/2 * (s_hat_1_RHLL * w1_n2 * (H_hat - qn_hat*c_hat) + s_hat_2_RHLL * w2_n2 * (.5*(u_hat*u_hat+v_hat*v_hat)) + s_hat_3_RHLL * w3_n2 * (H_hat+qn_hat*c_hat) + s_hat_4_RHLL * w4_n2 * (qt_hat))

        vel_Mass[cell0] = vel_Mass[cell0] - flux_Mass * faceArea[i] / cell_volumes[cell0]
        vel_Momx[cell0] = vel_Momx[cell0] - flux_Momx * faceArea[i] / cell_volumes[cell0]
        vel_Momy[cell0] = vel_Momy[cell0] - flux_Momy * faceArea[i] / cell_volumes[cell0]
        vel_Energy[cell0] = vel_Energy[cell0] - flux_Energy * faceArea[i] / cell_volumes[cell0]


    return vel_Mass, vel_Momx, vel_Momy, vel_Energy


@njit(nb.types.Tuple((float64[:],float64[:],float64[:],float64[:]))(float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64,int64,float64[:,:],float64[:],float64,float64))
def updateState(edge_data, Mass_n, Momx_n, Momy_n, Energy_n, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1):
    edge_index = edge_data[:,:2]
    d0d1 = edge_data[:,2:4]
    n_hat = edge_data[:,4:6]
    faceArea = edge_data[:,6]
    edge_centers = edge_data[:,7:8]

    rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
    n_edges = np.shape(edge_index)[0]

    N = np.shape(cell_centroids)[0]

    vel_Mass_c, vel_Momx_c, vel_Momy_c, vel_Energy_c = getVelocity(edge_data, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1)

    Mass_cph = Mass_c + dt*vel_Mass_c
    Momx_cph = Momx_c + dt*vel_Momx_c
    Momy_cph = Momy_c + dt*vel_Momy_c
    Energy_cph = Energy_c + dt*vel_Energy_c

    Mass_n = Mass_cph
    Momx_n = Momx_cph
    Momy_n = Momy_cph
    Energy_n = Energy_cph



    return Mass_n, Momx_n, Momy_n, Energy_n

def getGDLSPGsolution(x_init,N,edge_index,model,e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21):
        
    x_curr = np.zeros((N,4))

    x_hat_curr = torch.tensor(x_init,device=device)
    x_hat_next = torch.tensor(x_init,device=device)
    
    inputs = torch.zeros((1,N,4))
    
    
    x_next = model.decode(x_hat_next, inputs, 1, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)

    x_next = x_next[0,:,:]
    x_next = x_next

    x_hat_curr = x_hat_next.clone().to(device)
    x_curr = x_next.clone()
    x_curr = x_curr.detach().clone().type(torch.DoubleTensor).cpu().numpy()
    x_next = x_next.detach().clone().type(torch.DoubleTensor).numpy()

    x_next_state = np.zeros((4*N))

    Mass = x_curr[:,0]
    Momx = x_curr[:,1]
    Momy = x_curr[:,2]
    Energy = x_curr[:,3]
    x1 = torch.zeros((N,4))

    e_index = np.array(edge_index)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)
    edge_index = edge_index.to(device)

    x_curr = toState(Mass, Momx, Momy, Energy, N)

    f = np.zeros((4*N))
    lat_hist = torch.zeros((lat_dim,501)) 
    cond = 0.

    t_step = 0
    t_out = 0
    t_start = time.time()
    
    P_min = np.zeros(501)
    
    x_hist = np.zeros((4*N,501))
    iter_out = 0
    for t_step in range(501):
        lat_hist[:,t_step] = x_hat_curr.flatten().clone()
        del x_curr, x_hat_curr
        
        # for rom:
        x_hat_curr = torch.clone(x_hat_next)
        x_curr = torch.zeros((1,N,4),device=device)
        x_curr = model.decode(x_hat_curr,x_curr,1, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
        x_curr = x_curr[0,:,:].detach().type(torch.DoubleTensor).cpu().numpy()

        Mass_c = x_curr[:,0]
        Momx_c = x_curr[:,1]
        Momy_c = x_curr[:,2]
        Energy_c = x_curr[:,3]
        

        rho_c, vx_c, vy_c, P_c, E_c, H_c = getPrimitive(Mass_c, Momx_c, Momy_c, Energy_c, gamma, cell_volumes)
        
        
        x_hist[:N,t_step] = Mass_c
        x_hist[N:2*N,t_step] = Momx_c
        x_hist[2*N:3*N,t_step] = Momy_c
        x_hist[3*N:,t_step] = Energy_c
        
        Mass_n = np.zeros(N)
        Momx_n = np.zeros(N)
        Momy_n = np.zeros(N)
        Energy_n = np.zeros(N)
        
        
        Mass_n, Momx_n, Momy_n, Energy_n = updateState(edge_data, Mass_n, Momx_n, Momy_n, Energy_n, Mass_c, Momx_c, Momy_c, Energy_c, gamma, N, cell_centroids, cell_volumes, dt, u_1)
        
        
        k=0
        iter_tol = 1e5
        error1 = 1e5
        lambda0 = 1e-6
        iter_count = 0
        alpha = 1.
        
        J_decoder = torch.zeros((4*N,lat_dim))
        
        while True: 
            if iter_count==0:
                lambda_k = lambda0
            
            x_next_torch = torch.zeros((1,N,4),device=device)
            x_next_torch = model.decode(x_hat_next,x_next_torch,1, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21).detach()
            x_next_torch = x_next_torch[0,:,:]
            
            x_next_state[:N] = x_next_torch[:,0].detach().cpu().numpy()
            x_next_state[N:2*N] = x_next_torch[:,1].detach().cpu().numpy()
            x_next_state[2*N:3*N] = x_next_torch[:,2].detach().cpu().numpy()
            x_next_state[3*N:] = x_next_torch[:,3].detach().cpu().numpy()
            
            f = getResidual(x_next_state,Mass_n,Momx_n,Momy_n,Energy_n,N)
            error_HDM = np.linalg.norm(f)
            
            inputs = torch.zeros((1,N,4),device=device)
            J_decoder_temp = (torch.func.jacfwd(model.decode, argnums=0)(x_hat_next,inputs,1,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5,ai_54,ai_43,ai_32,ai_21)).detach()
            J_decoder[:N,:] = J_decoder_temp[0,:,0,0,:]
            J_decoder[N:2*N,:] = J_decoder_temp[0,:,1,0,:]
            J_decoder[2*N:3*N,:] = J_decoder_temp[0,:,2,0,:]
            J_decoder[3*N:4*N,:] = J_decoder_temp[0,:,3,0,:]
            J_decoder = J_decoder.to(device)
            
            ####### GN METHOD ###############
            
            B = torch.matmul(J_decoder.T,torch.tensor(f).type(torch.FloatTensor).to(device))
            C = torch.matmul(J_decoder.T,J_decoder)
            
            
            if torch.any(torch.isnan(B)):
                print('DID NOT CONVERGE!!!!!!!')
                lat_hist[:,t_step:] = torch.nan*torch.ones((lat_hist[:,t_step:]).size())
                return lat_hist
            
            p = -torch.linalg.solve(C,B)
            x_hat_next_old = x_hat_next.clone()
            
            error_LDM = torch.linalg.norm(B)
            if iter_count==0 and t_step==0:
                tol = 1e-3*error_LDM
                
            print('error_LDM: ', error_LDM, ', tol: ', tol, ', alpha: ', alpha)
            if error_LDM<tol:
                break
            
            
            x_hat_next = x_hat_next_old + alpha*p
            iter_count += 1
            if iter_count == 30:
                print('DID NOT CONVERGE!!!!!!!')
                lat_hist[:,t_step:] = torch.nan*torch.ones((lat_hist[:,t_step:]).size())
                return lat_hist
            ######## END OF GN METHOD ##########
    
            
    return lat_hist
            

if __name__=="__main__":
    lat_dim = int(sys.argv[1])
    ## Import mesh data
    mesh_file = 'Cyl_pooling_unpooling_normalized/'
    edge_index = np.load(mesh_file + 'edge_index.npy') ##
    cell_centroids = np.load(mesh_file + 'cell_centroids.npy')
    n_hat = np.load(mesh_file + 'n_hat.npy') ##
    faceArea = np.load(mesh_file + 'faceArea.npy') ##
    cell_volumes = np.load(mesh_file + 'cell_volumes.npy')

    d0d1 = np.zeros((np.shape(edge_index)[0],2))
    edge_centers = np.zeros((np.shape(edge_index)[0],2))
    
    latDimToDucros = [2, 4, 6]
    testSolsToDucros = [0, 3, 6, 9]
    

    cell_centroids = np.float64(cell_centroids)
    d0d1 = np.float64(d0d1)
    n_hat = np.float64(n_hat)
    faceArea = np.float64(faceArea)
    edge_centers = np.float64(edge_centers)
    cell_volumes = np.float64(cell_volumes)
    

    edge_data = np.concatenate((edge_index, d0d1), axis=1)
    edge_data = np.concatenate((edge_data, n_hat), axis=1)
    edge_data = np.concatenate((edge_data, np.reshape(faceArea, (np.size(faceArea),1))), axis=1)
    edge_data = np.concatenate((edge_data, edge_centers), axis=1)
    edge_data = edge_data[edge_data[:, 0].argsort()]


    v_1                    = 0.
    dt                     = 1e-3
    output_directory       = '../2D_outputs/Cyl_GNN_lat'+str(lat_dim) + '_mach_1125/'
    filename               = 'output'
    N                      = np.shape(cell_centroids)[0]
    gamma                  = 1.4
    train_file             = "output"
    mesh_location          = "Cyl_pooling_unpooling_normalized"
    volume_file            = "cell_volumes.npy"
    edge_index_file        = "edge_index.npy"
    
    model = architecture()
    model = torch.load('Cyl_GNN_lat_'+str(lat_dim) + '_500TS').cpu()

    model = model.eval()
    device = torch.device('cuda')
    model = model.to(device)


    directory = 'Cyl_pooling_unpooling_normalized/'
    e1 = torch.load(directory + 'edge_index').to(device)
    e2 = torch.load(directory + 'e2').to(device)
    e3 = torch.load(directory + 'e3').to(device)
    e4 = torch.load(directory + 'e4').to(device)
    e5 = torch.load(directory + 'e5').to(device)

    s1 = torch.load(directory + 's1').to(device)
    s2 = torch.load(directory + 's2').to(device)
    s3 = torch.load(directory + 's3').to(device)
    s4 = torch.load(directory + 's4').to(device)

    u1 = torch.load(directory + 'unpool1').to(device)
    u2 = torch.load(directory + 'unpool2').to(device)
    u3 = torch.load(directory + 'unpool3').to(device)
    u4 = torch.load(directory + 'unpool4').to(device)
    
    pos1 = torch.load(directory + 'pos1')
    pos2 = torch.load(directory + 'pos2')
    pos3 = torch.load(directory + 'pos3')
    pos4 = torch.load(directory + 'pos4')
    pos5 = torch.load(directory + 'pos5')


    ai_54 = knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_43 = knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_32 = knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)
    ai_21 = knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)    
    
    ai_54 = ai_54.to(device)
    ai_43 = ai_43.to(device)
    ai_32 = ai_32.to(device)
    ai_21 = ai_21.to(device)
    
    pos1 = pos1.to(device)
    pos2 = pos2.to(device)
    pos3 = pos3.to(device)
    pos4 = pos4.to(device)
    pos5 = pos5.to(device)
    

    u_all = [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95]
    
    numRep = 1
    
    x_hat_hist_full = torch.zeros((9,lat_dim,501))
    x_approx = torch.zeros((501,N,4))
    errors = np.zeros(10)
    totalTime = np.zeros(10)
    lowDimSolTime = np.zeros(10)
    decodeTime = np.zeros(10)
    encodeTime = np.zeros(10)
    
    
    
    for i in range(10):
        x_hist_in = np.load('Cyl_x_hist_500TS_'+str(1050+100*i)+'.npy')
        u_1 = u_all[i]
        
        for rep in range(numRep):
            t1 = time.time()
            rho, vx, vy, P = genDomain(cell_centroids,gamma,u_1,v_1)
            Mass, Mom_x, Mom_y, Energy, E, H = getConserved(rho, vx, vy, P, gamma, cell_volumes)
            x_init = torch.zeros((1,N,4))
            x_init[0,:,0] = torch.tensor(Mass)
            x_init[0,:,1] = torch.tensor(Mom_x)
            x_init[0,:,2] = torch.tensor(Mom_y)
            x_init[0,:,3] = torch.tensor(Energy)
            x_init = x_init.to(device)
            x_hat_init = (model.encode(x_init, 1, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach()
            t2 = time.time()
            encodeTime[i] += t2-t1
            
            t1 = time.time()
            x_hat_hist = getGDLSPGsolution(x_hat_init,N,edge_index,model,e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
            t2 = time.time()
            lowDimSolTime[i] += t2-t1
            
            inputs = torch.zeros((501,N,4),device=device)
            x_hat_hist = x_hat_hist.to(device)
            
            eBatSize = 501
            numEbat1= int(501/eBatSize)
            dBatSize = 501
            numDbat= int(501/dBatSize)
            
            
            t1 = time.time()
            for dBat in range(numDbat):
                inputs = torch.zeros((dBatSize,N,4),device=device)
                x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:] = (model.decode(x_hat_hist[:,dBat*dBatSize:(dBat+1)*dBatSize].T, inputs, dBatSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach().cpu()
                
            t2 = time.time()
            decodeTime[i] += t2-t1
            
        
        x_hist_approx = np.zeros((4*N,501))
        x_hist_approx[:N,:] = (x_approx[:,:,0]).T.cpu().detach().numpy()
        x_hist_approx[N:2*N,:] = (x_approx[:,:,1]).T.cpu().detach().numpy()
        x_hist_approx[2*N:3*N,:] = (x_approx[:,:,2]).T.cpu().detach().numpy()
        x_hist_approx[3*N:,:] = (x_approx[:,:,3]).T.cpu().detach().numpy()

        errors[i] = np.linalg.norm(x_hist_approx-x_hist_in[:,:],'fro')/np.linalg.norm(x_hist_in[:,:],'fro')
        print(np.linalg.norm(x_hist_approx-x_hist_in[:,:],'fro')/np.linalg.norm(x_hist_in[:,:],'fro'))


    np.save('Cyl_solutions/Cyl_GDLSPG_unconstrained_'+str(lat_dim)+'_errors_',errors)
    
    
    