import numpy as np
import sys
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy
from scipy import stats

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
    def __init__(self,n):
        super().__init__()

        self.conv1_1 = torch_geometric.nn.SAGEConv(2,16,aggr='mean',project=False,bias=False)
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
        
        self.layer5 = torch.nn.Linear(512,n)
        nn.init.xavier_uniform_(self.layer5.weight.data)

        self.layer6 = torch.nn.Linear(n,512)
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


        self.conv10_2 = torch_geometric.nn.SAGEConv(16,2,aggr='mean',project=False,bias=False)
        nn.init.xavier_uniform_(self.conv10_2.lin_l.weight)
        nn.init.xavier_uniform_(self.conv10_2.lin_r.weight)




    def encode(self,x,batchSize,e1,e2,e3,e4,e5,s1,s2,s3,s4,labels1,labels2,labels3,labels4):

        x = torch.nn.functional.elu(self.conv1_1(x,e1))
        x = torch.nn.functional.elu(self.conv1_2(x,e1))
        x = _avg_pool_x(labels1,x)

        x = torch.nn.functional.elu(self.conv2_1(x,e2))
        x = torch.nn.functional.elu(self.conv2_2(x,e2))
        x = _avg_pool_x(labels2,x)
        
        x = torch.nn.functional.elu(self.conv3_1(x,e3))
        x = torch.nn.functional.elu(self.conv3_2(x,e3))
        x = _avg_pool_x(labels3,x)

        x = torch.nn.functional.elu(self.conv4_1(x,e4))
        x = torch.nn.functional.elu(self.conv4_2(x,e4))
        x = _avg_pool_x(labels4,x)

        x = x.reshape((batchSize,512))
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
        

        x = x.reshape((batchSize,25600,2))

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

def getLasdiSolBE(C,A,F,x_init,N_lat,Nt,dt):
    tol = 1e-6
    x_hat_opInf = np.zeros((N_lat,Nt))
    x_hat_opInf[:,0] = x_init
    
    
    for t in range(Nt-1):
        x_hat_curr = x_hat_opInf[:,t]
        x_hat_next = x_hat_opInf[:,t]
        
        iterCount = 0
        while True:
            x_hat_sq = np.zeros((int(N_lat*(N_lat+1)/2)))
            dsq_dxhat = np.zeros((int(N_lat*(N_lat+1)/2),N_lat))
            count = 0
            for j in range(N_lat):
                for k in range(j+1):
                    x_hat_sq[count] = x_hat_next[j]*x_hat_next[k]
                    dsq_dxhat[count,j] += x_hat_next[k]
                    dsq_dxhat[count,k] += x_hat_next[j]
                    count += 1
            
            r = x_hat_next - x_hat_curr - dt*(C + A@x_hat_next + F@x_hat_sq)
            J = np.eye(N_lat) - dt*(A + F@dsq_dxhat)
            
            
            if np.linalg.norm(r)>tol:
                x_hat_next = x_hat_next - np.linalg.solve(J,r)
                iterCount+=1
                if iterCount==100:
                    print('failed')
                    return x_hat_opInf
            else:
                x_hat_opInf[:,t+1] = x_hat_next
                break
                
            
        
    return x_hat_opInf





N_lat = int(sys.argv[1])

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

testSol = 5


latDimToCanny = [3, 5, 7]
testSolsToCanny = [0, 3, 12, 15]

lamb0_range = [-8,0,-8,-8,-1]
lamb1_range = [-8,2,3,3,1]

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

x_hist_in = np.load('Kobayashi_x_hist_imp_May5_full.npy')
x_hist_in_test = np.load('Kobayashi_x_hist_imp_May5_full_test.npy')

            

x_hist_in = x_hist_in[:,:,:Nt]
x_hist_in_test = x_hist_in_test[:,:,:Nt]

nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((2*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]
    
directory = '2D_pooling_unpooling_Kobayashi_March18/'

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

labels1 = (torch.load(directory + 'labels1')).to(cuda)
labels2 = (torch.load(directory + 'labels2')).to(cuda)
labels3 = (torch.load(directory + 'labels3')).to(cuda)
labels4 = (torch.load(directory + 'labels4')).to(cuda)

ai_54 = (knn(pos5.cpu(), pos4.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_43 = (knn(pos4.cpu(), pos3.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_32 = (knn(pos3.cpu(), pos2.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)
ai_21 = (knn(pos2.cpu(), pos1.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1)).to(cuda)

pos1 = pos1.to(cuda)
pos2 = pos2.to(cuda)
pos3 = pos3.to(cuda)
pos4 = pos4.to(cuda)
pos5 = pos5.to(cuda)



model = torch.load('Kobayashi_May5_full_' + str(N_lat))

x_hist_in = torch.tensor(x_hist_in)

x_hat_train = torch.zeros((nSol,N_lat,Nt),device=cuda)
x_hat_dot_train = torch.zeros((nSol,N_lat,Nt),device=cuda)
D = torch.zeros((1+N_lat+int((N_lat+1)*N_lat/2), (Nt)*nSol))
R = torch.zeros((N_lat, (Nt)*nSol))

eBatSize = 50
numEbat = int(Nt/eBatSize)
            
# new code using higher order approximations to remove noise
for sol in range(nSol):
    x_hist = torch.zeros((Nt,N,2),device=cuda)
    x_hist[:,:,0] = (x_hist_in[sol,:N,:].T).to(cuda)
    x_hist[:,:,1] = (x_hist_in[sol,N:,:].T).to(cuda)
    
    for eBat in range(numEbat):
        x_hat_train[sol,:,eBat*eBatSize:(eBat+1)*eBatSize] = (model.encode(x_hist[eBat*eBatSize:(eBat+1)*eBatSize,:,:], eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)).detach().T
    x_hat_train[sol,:,numEbat*eBatSize:] = (model.encode(x_hist[numEbat*eBatSize:,:,:], Nt-numEbat*eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)).detach().T
    
    
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

lamb = np.zeros(2)


lamb[0] = lamb0_range[N_lat-3]
lamb[1] = lamb1_range[N_lat-3]

C,A,F = getLasdiOperators(lamb,D,R)

x_hist_in_test = torch.tensor(x_hist_in_test)
nSol = x_hist_in_test.size(dim=0)
x_hat_test = torch.zeros((nSol,N_lat,Nt))

x_hist_in = torch.tensor(x_hist_in)

for sol in range(nSol):
    x_hist_test = torch.zeros((Nt,N,2),device=cuda)
    x_hist_test[:,:,0] = (x_hist_in_test[sol,:N,:].T).to(cuda)
    x_hist_test[:,:,1] = (x_hist_in_test[sol,N:,:].T).to(cuda)
    
    for eBat in range(numEbat):
        x_hat_test[sol,:,eBat*eBatSize:(eBat+1)*eBatSize] = (model.encode(x_hist_test[eBat*eBatSize:(eBat+1)*eBatSize,:,:], eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)).detach().T
    x_hat_test[sol,:,numEbat*eBatSize:] = (model.encode(x_hist_test[numEbat*eBatSize:,:,:], Nt-numEbat*eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)).detach().T
    

errors = np.zeros(nSol)
totalTime = np.zeros(nSol)
lowDimSolTime = np.zeros(nSol)
decodeTime = np.zeros(nSol)
encodeTime = np.zeros(nSol)

nRep = 10

x_hist_in_test = x_hist_in_test.detach().cpu().numpy()
x_hist_in = x_hist_in.detach().cpu().numpy()

a_all = [.3875, .3875, .3875, .3875, .4625, .4625, .4625, .4625, .5375, .5375, .5375, .5375, .6125, .6125, .6125, .6125]
b_all = [.3875, .4625, .5375, .6125, .3875, .4625, .5375, .6125, .3875, .4625, .5375, .6125, .3875, .4625, .5375, .6125]
    

for sol in range(nSol):
    for rep in range(nRep):
        ts = time.time()
        t1 = time.time()
        a = a_all[sol]
        b = b_all[sol]
        
        r0 = .075
        r = np.sqrt(np.square(coords[:,0]-a) + np.square(coords[:,1]-b))
        phi_init = .5*(1 - np.tanh((r-r0)/(2*eps*np.sqrt(2))))
        U_init = Delta*phi_init
        x_init = torch.zeros((1,N,2),device=cuda)
        x_init[0,:,0] = torch.tensor(phi_init,dtype=torch.float32).to(cuda)
        x_init[0,:,1] = torch.tensor(U_init,dtype=torch.float32).to(cuda)
    
        x_hat_init = model.encode(x_init, 1, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4).detach().cpu().flatten().numpy()
        t2 = time.time()
        encodeTime[sol] += t2-t1
    
        x_init = x_hat_test[sol,:,0]
        t1 = time.time()
        x_hat_opInf = getLasdiSolBE(C,A,F,x_init,N_lat,Nt,dt)
        t2 = time.time()
        lowDimSolTime[sol] += t2-t1
        x_hat_opInf = torch.tensor(x_hat_opInf,device=cuda,dtype=torch.float32)
    
        x_approx = torch.zeros((Nt,N,2))
        dBatSize = 50
        numDbat = int(Nt/dBatSize)
        
        t3 = time.time()
        x_hat_train = torch.tensor(x_hat_train,device=torch.device('cuda'))
        
        
        x_approx = model.decode(x_hat_opInf[:,:].T, Nt, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21).detach().cpu()

        t4 = time.time()
        decodeTime[sol] += t4-t3
        x_approx = x_approx.detach().cpu()
        x_hat_train = x_hat_train.cpu()
        
        x_hist_approx = np.zeros((2*N,Nt))
        x_hist_approx[:N,:] = (x_approx[:,:,0]).T
        x_hist_approx[N:,:] = (x_approx[:,:,1]).T

        te = time.time()
        totalTime[sol] += te-ts
        
        if sol in testSolsToCanny and N_lat in latDimToCanny:
            np.save('Kobayashi_solutions/Kobayashi_GNN_LaSDI_'+str(N_lat) + '_testSol_'+str(sol), x_hat_opInf.detach().cpu().numpy())
            
        
    errors[sol] = np.linalg.norm(x_hist_approx[:,:]-x_hist_in_test[sol,:,:],'fro')/np.linalg.norm(x_hist_in_test[sol,:,:],'fro')
    

print(errors)
np.save('wallclock_GNN_La_solutions/GNN_LaSDI_'+str(N_lat)+'_errors_',errors)
np.save('Kobayashi_solutions/GNN_LaSDI_'+str(N_lat)+'_lowDimSolTime_',lowDimSolTime/nRep)
np.save('Kobayashi_solutions/GNN_LaSDI_'+str(N_lat)+'_decodeTime_',decodeTime/nRep)
np.save('Kobayashi_solutions/GNN_LaSDI_'+str(N_lat)+'_encodeTime_',encodeTime/nRep)
np.save('Kobayashi_solutions/GNN_LaSDI_'+str(N_lat)+'_totalTime_',totalTime/nRep)
