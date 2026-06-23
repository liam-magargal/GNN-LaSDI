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


x_hist_in = np.zeros((nSol,4*N,Nt))
x_hist_in_test = np.zeros((nSol-1,4*N,Nt))


for i in range(nSol):
    output_location = 'Cyl_x_hist_500TS_'+str(1000+100*i)+'.npy' #training data
    volume_file = 'cell_volumes.npy'
    x_hist_temp = (np.load(output_location))
    x_hist_in[i,:N,:] = x_hist_temp[:N,:]
    x_hist_in[i,N:2*N,:] = x_hist_temp[N:2*N,:]
    x_hist_in[i,2*N:3*N,:] = x_hist_temp[2*N:3*N,:]
    x_hist_in[i,3*N:,:] = x_hist_temp[3*N:,:]
    


for i in range(nSol-1):
    output_location = 'Cyl_x_hist_500TS_test_'+str(1050+100*i)+'.npy' #test data
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
    

print(np.shape(x_hist))

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

pos1 = (torch.load(directory + 'pos1'))
pos2 = (torch.load(directory + 'pos2'))
pos3 = (torch.load(directory + 'pos3'))
pos4 = (torch.load(directory + 'pos4'))
pos5 = (torch.load(directory + 'pos5'))


ai_54 = (knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_43 = (knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_32 = (knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_21 = (knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)

pos1 = pos1.to(cuda)
pos2 = pos2.to(cuda)
pos3 = pos3.to(cuda)
pos4 = pos4.to(cuda)
pos5 = pos5.to(cuda)

model = torch.load('Cyl_GNN_lat_'+str(N_lat) + '_500TS')
cell_volumes = np.load(directory + 'cell_volumes.npy')

x_hist_in = torch.tensor(x_hist_in)

latDimToDucros = [2, 4, 5, 6]
testSolsToDucros = [0, 3, 5, 6, 9]


x_hat_train = torch.zeros((nSol,N_lat,Nt),device=cuda)
x_hat_dot_train = torch.zeros((nSol,N_lat,Nt),device=cuda)
D = torch.zeros((1+N_lat+int((N_lat+1)*N_lat/2), (Nt)*nSol))
R = torch.zeros((N_lat, (Nt)*nSol))

eBatSize = 501
numEbat = int(Nt/eBatSize)


for sol in range(nSol):
    x_hist = torch.zeros((Nt,N,4),device=cuda)
    x_hist[:,:,0] = (x_hist_in[sol,:N,:].T).to(cuda)
    x_hist[:,:,1] = (x_hist_in[sol,N:2*N,:].T).to(cuda)
    x_hist[:,:,2] = (x_hist_in[sol,2*N:3*N,:].T).to(cuda)
    x_hist[:,:,3] = (x_hist_in[sol,3*N:,:].T).to(cuda)
    
    for eBat in range(numEbat):
        x_hat_train[sol,:,eBat*eBatSize:(eBat+1)*eBatSize] = (model.encode(x_hist[eBat*eBatSize:(eBat+1)*eBatSize,:,:], eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach().T
    
    
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


D = D.numpy()
R = R.numpy()
x_hat_train = x_hat_train.cpu().numpy()

allLamb0 = [1,2,-8,-6,2]
allLamb1 = [4,3,3,4,4]


lamb = np.zeros(2)

lamb[0] = allLamb0[N_lat-2]
lamb[1] = allLamb1[N_lat-2]

errors = getLasdiError(lamb,D,R,x_hat_train,N_lat,Nt,dt)

C,A,F = getLasdiOperators(lamb,D,R)


x_hist_in_test = torch.tensor(x_hist_in_test)
nSol = x_hist_in_test.size(dim=0)
x_hat_test = torch.zeros((nSol,N_lat,Nt))

x_hist_in = torch.tensor(x_hist_in)

for sol in range(nSol):
    x_hist_test = torch.zeros((Nt,N,4),device=cuda)
    x_hist_test[:,:,0] = (x_hist_in_test[sol,:N,:].T).to(cuda)
    x_hist_test[:,:,1] = (x_hist_in_test[sol,N:2*N,:].T).to(cuda)
    x_hist_test[:,:,2] = (x_hist_in_test[sol,2*N:3*N,:].T).to(cuda)
    x_hist_test[:,:,3] = (x_hist_in_test[sol,3*N:,:].T).to(cuda)
    
    for eBat in range(numEbat):
        x_hat_test[sol,:,eBat*eBatSize:(eBat+1)*eBatSize] = (model.encode(x_hist_test[eBat*eBatSize:(eBat+1)*eBatSize,:,:], eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach().T
    

errors = np.zeros(nSol)
x_hist_in_test = x_hist_in_test.detach().cpu().numpy()
x_hist_in = x_hist_in.detach().cpu().numpy()

numRep = 10

lowDimSolTime = np.zeros(nSol)
decodeTime = np.zeros(nSol)
encodeTime = np.zeros(nSol)
totalTime = np.zeros(nSol)
for sol in range(nSol):
    print(sol)
    for rep in range(numRep):
        ts = time.time()
        t1 = time.time()
        rho, u, v, P =  genDomain(pos1,gamma,1.05+0.1*sol,0)
        Mass, Mom_x, Mom_y, Energy, E, H = getConserved( rho, u, v, P, gamma, cell_volumes, N )
        x_init = torch.zeros((1,N,4))
        x_init[0,:,0] = torch.tensor(Mass)
        x_init[0,:,1] = torch.tensor(Mom_x)
        x_init[0,:,2] = torch.tensor(Mom_y)
        x_init[0,:,3] = torch.tensor(Energy)
        x_init = x_init.to(cuda)
        x_hat_init = (model.encode(x_init, 1, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach().cpu().flatten().numpy()
        t2 = time.time()
        encodeTime[sol] += t2-t1
        
        t1 = time.time()
        x_hat_opInf = getLasdiSol(C,A,F,x_hat_init,N_lat,Nt,dt)
        t2 = time.time()
        lowDimSolTime[sol] += t2-t1
    
        x_hat_opInf = torch.tensor(x_hat_opInf,device=cuda,dtype=torch.float32)
        x_approx = torch.zeros((Nt,N,4))
        dBatSize = 501
        numDbat = int(Nt/dBatSize)
        
        t1 = time.time()
        x_hat_train = torch.tensor(x_hat_train,device=torch.device('cuda'))
        for dBat in range(numDbat):
            x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:] = (model.decode(x_hat_opInf[:,dBat*dBatSize:(dBat+1)*dBatSize].T, x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:], dBatSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach().cpu()
        t2 = time.time()
        decodeTime[sol] += t2-t1
        
        te = time.time()
        totalTime[sol] += te-ts
    x_approx = x_approx.detach().cpu()
    x_hat_train = x_hat_train.cpu()
    
    x_hist_approx = np.zeros((4*N,Nt))
    x_hist_approx[:N,:] = (x_approx[:,:,0]).T
    x_hist_approx[N:2*N,:] = (x_approx[:,:,1]).T
    x_hist_approx[2*N:3*N,:] = (x_approx[:,:,2]).T
    x_hist_approx[3*N:,:] = (x_approx[:,:,3]).T
    
    
    x_hat_opInf = x_hat_opInf.detach().cpu().numpy()
    
    if sol in testSolsToDucros and N_lat in latDimToDucros:
        np.save('Cyl_solutions/Cyl_GNN_LaSDI_unconstrained_'+str(N_lat) + '_testSol_'+str(sol), x_hat_opInf)
    
    errors[sol] = np.linalg.norm(x_hist_approx[:,:]-x_hist_in_test[sol,:,:],'fro')/np.linalg.norm(x_hist_in_test[sol,:,:],'fro')
    

np.save('Cyl_solutions/Cyl_GNN_LaSDI_unconstrained_'+str(N_lat)+'_errors_',errors)
np.save('Cyl_solutions/Cyl_GNN_LaSDI_unconstrained_'+str(N_lat)+'_lowDimSolTime_',lowDimSolTime/numRep)
np.save('Cyl_solutions/Cyl_GNN_LaSDI_unconstrained_'+str(N_lat)+'_decodeTime_',decodeTime/numRep)
np.save('Cyl_solutions/Cyl_GNN_LaSDI_unconstrained_'+str(N_lat)+'_encodeTime_',encodeTime/numRep)
np.save('Cyl_solutions/Cyl_GNN_LaSDI_unconstrained_'+str(N_lat)+'_totalTime_',totalTime/numRep)
