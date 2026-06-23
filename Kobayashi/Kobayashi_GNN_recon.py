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

x_hist_in = np.load('Kobayashi_x_hist_imp_May5_full_test.npy')


nSol = np.shape(x_hist_in)[0]

x_hist = np.zeros((2*N,nSol*Nt))

for i in range(nSol):
    x_hist[:,i*Nt:(i+1)*Nt] = x_hist_in[i,:,:]
    
directory = '2D_pooling_unpooling_Kobayashi_March18/'


cuda = torch.device('cuda')

# load graph hierarchy
e1 = torch.load(directory + 'edge_index')
e2 = torch.load(directory + 'e2')
e3 = torch.load(directory + 'e3')
e4 = torch.load(directory + 'e4')
e5 = torch.load(directory + 'e5')
    
s1 = torch.load(directory + 's1')
s2 = torch.load(directory + 's2')
s3 = torch.load(directory + 's3')
s4 = torch.load(directory + 's4')
    
u1 = torch.load(directory + 'unpool1')
u2 = torch.load(directory + 'unpool2')
u3 = torch.load(directory + 'unpool3')
u4 = torch.load(directory + 'unpool4')
    
pos1 = torch.load(directory + 'pos1')
pos2 = torch.load(directory + 'pos2')
pos3 = torch.load(directory + 'pos3')
pos4 = torch.load(directory + 'pos4')
pos5 = torch.load(directory + 'pos5')
    
labels1 = torch.load(directory + 'labels1')
labels2 = torch.load(directory + 'labels2')
labels3 = torch.load(directory + 'labels3')
labels4 = torch.load(directory + 'labels4')
    
ai_54 = knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)
ai_43 = knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)
ai_32 = knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)
ai_21 = knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)
    

e1 = e1.to(cuda)
e2 = e2.to(cuda)
e3 = e3.to(cuda)
e4 = e4.to(cuda)
e5 = e5.to(cuda)
s1 = s1.to(cuda)
s2 = s2.to(cuda)
s3 = s3.to(cuda)
s4 = s4.to(cuda)
u1 = u1.to(cuda)
u2 = u2.to(cuda)
u3 = u3.to(cuda)
u4 = u4.to(cuda)
pos1 = pos1.to(cuda)
pos2 = pos2.to(cuda)
pos3 = pos3.to(cuda)
pos4 = pos4.to(cuda)
pos5 = pos5.to(cuda)
ai_54 = ai_54.to(cuda)
ai_43 = ai_43.to(cuda)
ai_32 = ai_32.to(cuda)
ai_21 = ai_21.to(cuda)
labels1 = labels1.to(cuda)
labels2 = labels2.to(cuda)
labels3 = labels3.to(cuda)
labels4 = labels4.to(cuda)


for i in range(5):
    N_lat = 3+i
    reconError = np.zeros(nSol)
    model = torch.load('Kobayashi_May5_full_' + str(N_lat))
    
    x_hist_in = torch.tensor(x_hist_in)
    
    
    x_hat_test = torch.zeros((nSol,N_lat,Nt),device=cuda)
    eBatSize = 50
    numEbat = int(Nt/eBatSize)
    
    for sol in range(nSol):
        x_hist = torch.zeros((Nt,N,2),device=cuda)
        x_hist[:,:,0] = (x_hist_in[sol,:N,:].T).to(cuda)
        x_hist[:,:,1] = (x_hist_in[sol,N:,:].T).to(cuda)
        
        for eBat in range(numEbat):
            x_hat_test[sol,:,eBat*eBatSize:(eBat+1)*eBatSize] = (model.encode(x_hist[eBat*eBatSize:(eBat+1)*eBatSize,:,:], eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)).detach().T
        start = eBatSize*numEbat
        end = Nt
        x_hat_test[sol,:,start:end] = (model.encode(x_hist[start:end,:,:], end-start, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)).detach().T
        
        x_approx = torch.zeros((Nt,N,2))
        dBatSize = 50
        numDbat = int(Nt/dBatSize)
        
        x_hat_test = torch.tensor(x_hat_test,device=torch.device('cuda'))
        for dBat in range(numDbat):
            x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:] = (model.decode(x_hat_test[sol,:,dBat*dBatSize:(dBat+1)*dBatSize].T, dBatSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach()
        start = dBatSize*numDbat
        end = Nt
        x_approx[start:end,:,:] = (model.decode(x_hat_test[sol,:,start:end].T, end-start, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach()
        
        x_approx = x_approx.detach().cpu()
        x_hat_test = x_hat_test.cpu()
        
        x_hist_approx = np.zeros((2*N,Nt))
        x_hist_approx[:N,:] = (x_approx[:,:,0]).T.detach().numpy()
        x_hist_approx[N:,:] = (x_approx[:,:,1]).T.detach().numpy()
   
        x_hist_in = x_hist_in.detach().cpu().numpy()
        reconError[sol] = np.linalg.norm(x_hist_approx[:,:]-x_hist_in[sol,:,:],'fro')/np.linalg.norm(x_hist_in[sol,:,:],'fro')
        print(reconError[sol])
        x_hist_in = torch.tensor(x_hist_in)
        
    np.save('Kobayashi_solutions/Kobayashi_GNN_recon_'+str(N_lat),reconError)
