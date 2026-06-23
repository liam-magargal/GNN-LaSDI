import numpy as np
import sys
import time

import torch
import torch.nn as nn
import torch_geometric

from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from typing import Callable, Optional, Tuple
from torch import Tensor

from numba import njit, jit, prange, float64, int64, float32
import numba as nb

from scipy.spatial.distance import directed_hausdorff


def _avg_pool_x(
    cluster: Tensor,
    x: Tensor,
    size: Optional[int] = None,
) -> Tensor:
    return scatter(x, cluster, dim=1, reduce='mean')


def knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, assign_index: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.024testSol>`_ paper.
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
    
    
    def decode(self,x,y,batchSize,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5, ai_54, ai_43, ai_32, ai_21):
        
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
        x = torch.nn.functional.sigmoid(self.conv10_2(x,e1))
        xc = x.clone()

        y[:,:,0] = ((xc[:,:,0]) * 3.) + .5
        y[:,:,1] = (xc[:,:,1]) * 4.
        y[:,:,2] = (xc[:,:,2] * 6) - 3.
        y[:,:,3] = 1/2/((xc[:,:,0])*3. + .5)*((xc[:,:,1]*4)**2 + (xc[:,:,2]*6. - 3.)**2) + ((xc[:,:,3]) * 18. + 1.)

        y = y.reshape((batchSize,4148,4))
        
        return y

@jit(nb.types.Tuple((float32[:],float32[:],float32[:],float32[:],float32[:],float32[:]))(float32[:],float32[:],float32[:],float32[:],float64,float32[:]) )
def getPrimitive( Mass, Mom_x, Mom_y, Energy, gamma, vol ):
    N = np.shape(Mass)[0]
    P = np.zeros(N,dtype=np.float32)
    H = np.zeros(N,dtype=np.float32)
    rho = Mass
    u  = Mom_x / rho
    v  = Mom_y / rho
    E = Energy / rho
    P[:] = (E - .5*(u*u + v*v)) * (gamma-1)*rho
    H[:] = (gamma) / (gamma-1) * P / rho + .5*(u*u + v*v)
    
    return rho, u, v, P, E, H

@njit((float64)(float64,float64))
def np_max(val1, val2):
    if val1>val2:
        return val1
    else:
        return val2



@njit((float32[:])(float32[:,:],float32[:],float32[:],float32[:],float32[:],float64,int64,float32[:,:],float32[:],float64,float64))
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
    ducros_sensor = np.zeros((N),dtype=np.float32)
    
    
    ## using RHLL flux (Nishikawa 2008)
    for i in range(n_edges):
        cell0 = int(edge_index[i,0])
        cell1 = int(edge_index[i,1])
        
        rho_in = 1.
        vx_in = u_1*np.sqrt(1.4) #.25
        
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

    
    

# N_lat = 3
N_lat = int(sys.argv[1])
testSol = int(sys.argv[2])

print('GNN-LaSDI,', N_lat, testSol)

Nt = 501
dt = 1e-3
gamma = 1.4


output_location = 'Cyl_x_hist_500TS_'+str(1050+100*testSol)+'.npy'
x_hist_in = (np.load(output_location))
x_hist_in = np.float32(x_hist_in)

x_hat_hist_in = np.load('Cyl_solutions/Cyl_GNN_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'.npy')


nSol = np.shape(x_hist_in)[0]


directory = 'Cyl_pooling_unpooling_normalized/'


cuda = torch.device('cuda')

# load graph hierarchy
e1 = (torch.load(directory + 'edge_index')).to(cuda)
e2 = (torch.load(directory + 'e2')).to(cuda)
e3 = (torch.load(directory + 'e3')).to(cuda)
e4 = (torch.load(directory + 'e4')).to(cuda)
e5 = (torch.load(directory + 'e5')).to(cuda)

s1 = (torch.load(directory + 's1')).to(cuda)
s2 = (torch.load(directory + 's2')).to(cuda)
s3 = (torch.load(directory + 's3')).to(cuda)
s4 = (torch.load(directory + 's4')).to(cuda)

u1 = (torch.load(directory + 'unpool1')).to(cuda)
u2 = (torch.load(directory + 'unpool2')).to(cuda)
u3 = (torch.load(directory + 'unpool3')).to(cuda)
u4 = (torch.load(directory + 'unpool4')).to(cuda)

pos1 = (torch.load(directory + 'pos1')).to(cuda)
pos2 = (torch.load(directory + 'pos2')).to(cuda)
pos3 = (torch.load(directory + 'pos3')).to(cuda)
pos4 = (torch.load(directory + 'pos4')).to(cuda)
pos5 = (torch.load(directory + 'pos5')).to(cuda)


ai_54 = (knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_43 = (knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_32 = (knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_21 = (knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)


model = torch.load('Cyl_GNN_lat_'+str(N_lat) + '_500TS_constrained')

mesh_file = 'Cyl_pooling_unpooling_normalized/'
edge_index = np.load(mesh_file + 'edge_index.npy') ##
cell_centroids = np.load(mesh_file + 'cell_centroids.npy')
d0d1 = np.zeros((np.shape(edge_index)[0],2))
n_hat = np.load(mesh_file + 'n_hat.npy') ##
faceArea = np.load(mesh_file + 'faceArea.npy') ##
edge_centers = np.load(mesh_file + 'edge_centers.npy') ##
edge_centers = np.zeros((np.shape(edge_index)[0],2)) 
cell_volumes = np.load(mesh_file + 'cell_volumes.npy')

edge_index = np.float32(edge_index)
cell_centroids = np.float32(cell_centroids)
n_hat = np.float32(n_hat)
faceArea = np.float32(faceArea)
edge_centers = np.float32(edge_centers)
cell_volumes = np.float32(cell_volumes)


N = np.shape(cell_centroids)[0]

edge_data = np.concatenate((edge_index, d0d1), axis=1)
edge_data = np.concatenate((edge_data, n_hat), axis=1)
edge_data = np.concatenate((edge_data, np.reshape(faceArea, (np.size(faceArea),1))), axis=1)
edge_data = np.concatenate((edge_data, edge_centers), axis=1)
edge_data = edge_data[edge_data[:, 0].argsort()]

edge_data = np.float32(edge_data)



x_approx = torch.zeros((Nt,N,4))
dBatSize = 501
numDbat = int(Nt/dBatSize)


x_hat_hist_in = torch.tensor(x_hat_hist_in,device=torch.device('cuda'))
inputs = torch.zeros((N,))
for dBat in range(numDbat):
    x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:] = (model.decode(x_hat_hist_in[:,dBat*dBatSize:(dBat+1)*dBatSize].T, x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:], dBatSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach().cpu()

x_approx = x_approx.detach().cpu().numpy()

x_hist_approx = np.zeros((4*N,Nt))
x_hist_approx[:N,:] = (x_approx[:,:,0]).T
x_hist_approx[N:2*N,:] = (x_approx[:,:,1]).T
x_hist_approx[2*N:3*N,:] = (x_approx[:,:,2]).T
x_hist_approx[3*N:,:] = (x_approx[:,:,3]).T
    

t_hist = np.linspace(0,.25,Nt)


interfaceError = np.zeros((Nt))
interfaceError_2norm = np.zeros((Nt))
interfaceError_sum = np.zeros((Nt))



u_1 = 1 + 0.05*testSol
pos1 = pos1.cpu().numpy()
for t in range(Nt):
    t1 = time.time()
    Mass_c = x_approx[t,:,0]
    Momx_c = x_approx[t,:,1]
    Momy_c = x_approx[t,:,2]
    Energy_c = x_approx[t,:,3]
    t2 = time.time()
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
    t3 = time.time()
    
    
    if t==250:
        np.save('Cyl_solutions/GNN_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_ducros250',ducros_approx)
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_ducros250',ducros_GT)
        np.save('Cyl_solutions/GNN_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_sol250',x_hist_approx[:,t])
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_sol250',x_hist_in[:,t])
    if t==500:
        np.save('Cyl_solutions/GNN_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_ducros500',ducros_approx)
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_ducros500',ducros_GT)
        np.save('Cyl_solutions/GNN_LaSDI_'+str(N_lat)+'_testSol_'+str(testSol)+'_sol500',x_hist_approx[:,t])
        np.save('Cyl_solutions/GT_testSol_'+str(testSol)+'_sol500',x_hist_in[:,t])
        
        
    pointCloud_approx = pos1[np.where(ducros_approx>0),:]
    pointCloud_GT = pos1[np.where(ducros_GT>0),:]
    
    interfaceError[t] = max(directed_hausdorff(pointCloud_approx[0,:,:], pointCloud_GT[0,:,:])[0], directed_hausdorff(pointCloud_GT[0,:,:], pointCloud_approx[0,:,:])[0])
    

np.save('Cyl_solutions/GNN_LaSDI_pc_error_'+str(N_lat)+'_testSol_'+str(testSol),interfaceError)

