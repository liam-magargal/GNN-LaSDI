import numpy as np
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric

from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

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
    

def loadDataTotal(numTrain, numTest, numVal, ns, N, directory):
    x_hist = torch.zeros((numTrain*ns,N,4))
    x_test = torch.zeros((numTest*ns,N,4))
    gamma = 1.4
    
    for i in range(numTest):
        output_location = 'Cyl_x_hist_500TS_'+str(1050+100*i)+'.npy'
        volume_file = 'cell_volumes.npy'
        x_test_temp = torch.tensor(np.load(output_location))
        x_test[ns*i:ns*(i+1),:,0] = x_test_temp[:N,:].T
        x_test[ns*i:ns*(i+1),:,1] = x_test_temp[N:2*N,:].T
        x_test[ns*i:ns*(i+1),:,2] = x_test_temp[2*N:3*N,:].T
        x_test[ns*i:ns*(i+1),:,3] = x_test_temp[3*N:,:].T
        
    
    pos = None
    
    x_train = x_hist
    x_val = 0
    
    return x_train, x_val, x_test, pos


if __name__=="__main__":
    lat_dim = int(sys.argv[1])

    batchSize = 1

    cuda = torch.device('cuda')

    N = 4148
    ns = 501
    numTrain = 11
    numTest = 10
    numVal = 0
    filename = 'output'
    directory = None
    
    x_train, x_val, x_test, pos = loadDataTotal(numTrain, numTest, numVal, ns, N, directory)
    
    model = architecture()
    model = torch.load('Cyl_GNN_lat_'+str(lat_dim) + '_500TS_constrained')
    
    gamma = 1.4

    x_test = x_test.to(cuda)
    
    
    directory = 'Cyl_pooling_unpooling_normalized/'
    
    e1 = torch.load(directory + 'edge_index').to(cuda)
    e2 = torch.load(directory + 'e2').to(cuda)
    e3 = torch.load(directory + 'e3').to(cuda)
    e4 = torch.load(directory + 'e4').to(cuda)
    e5 = torch.load(directory + 'e5').to(cuda)
    
    s1 = torch.load(directory + 's1').to(cuda)
    s2 = torch.load(directory + 's2').to(cuda)
    s3 = torch.load(directory + 's3').to(cuda)
    s4 = torch.load(directory + 's4').to(cuda)
    
    u1 = torch.load(directory + 'unpool1').to(cuda)
    u2 = torch.load(directory + 'unpool2').to(cuda)
    u3 = torch.load(directory + 'unpool3').to(cuda)
    u4 = torch.load(directory + 'unpool4').to(cuda)
    
    pos1 = torch.load(directory + 'pos1').to(cuda)
    pos2 = torch.load(directory + 'pos2').to(cuda)
    pos3 = torch.load(directory + 'pos3').to(cuda)
    pos4 = torch.load(directory + 'pos4').to(cuda)
    pos5 = torch.load(directory + 'pos5').to(cuda)
    
    ai_54 = knn(pos5.cpu(), pos4.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(cuda)
    ai_43 = knn(pos4.cpu(), pos3.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(cuda)
    ai_32 = knn(pos3.cpu(), pos2.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(cuda)
    ai_21 = knn(pos2.cpu(), pos1.cpu(), k=3, batch_x=None, batch_y=None, num_workers=1).to(cuda)
   
    cell_volumes = np.load(directory + 'cell_volumes.npy')
    
    rel_error = torch.zeros((ns))
    
    
    x_out = torch.zeros((N,4))
    
    input_hist = torch.zeros(4*N,ns)
    pred_hist = torch.zeros(4*N,ns)
    
    lat_hist = torch.zeros((ns,lat_dim))
    
    num=0
    den=0

    eBatSize = 250
    numEbat1= int(ns*numTest/eBatSize)
    dBatSize = 250
    numDbat= int(ns*numTest/dBatSize)
    
    
    x_hat_hist = torch.zeros((lat_dim,ns*numTest),device=cuda)
    x_approx = torch.zeros((ns*numTest,N,4),device=cuda)
    x_preSig = torch.zeros((ns*numTest,N,4),device=cuda)
    x3_preSigmoid = torch.zeros((ns*numTest,N))
    J_decoder = torch.zeros(ns*numTest,lat_dim,N)
    
    for eBat in range(numEbat1):
        inputs = (x_test[eBat*eBatSize:(eBat+1)*eBatSize,:,:]).clone().to(cuda)
        x_hat_hist[:,eBat*eBatSize:(eBat+1)*eBatSize] = (model.encode(inputs, eBatSize, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach().T
    
    if numEbat1*eBatSize:
        start = numEbat1*eBatSize
        end = ns*numTest
        inputs = (x_test[start:end,:,:]).clone().to(cuda)
        x_hat_hist[:,start:end] = (model.encode(inputs, end-start, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach().T
        
    for dBat in range(numDbat):
        inputs = torch.zeros((dBatSize,N,4),device=cuda)
        x_approx[dBat*dBatSize:(dBat+1)*dBatSize,:,:], x_preSig[dBat*dBatSize:(dBat+1)*dBatSize,:,:] = (model.decode(x_hat_hist[:,dBat*dBatSize:(dBat+1)*dBatSize].T, inputs, dBatSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21))#.detach().cpu()

    if numDbat*dBatSize:
        start = numDbat*dBatSize
        end = ns*numTest
        inputs = (x_test[start:end,:,:]).clone().to(cuda)
        x_hat_hist[:,start:end] = (model.encode(inputs, end-start, e1, e2, e3, e4, e5, s1, s2, s3, s4)).detach().T
        x_approx[start:end,:,:], x_preSig[start:end,:,:] = (model.decode(x_hat_hist[:,start:end].T, inputs, end-start, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21))#.detach().cpu()
        
        

    errors = np.zeros(numTest)
    for i in range(numTest):
        x_hist = torch.zeros((4*N,ns))
        x_hist[:N,:] = x_test[i*ns:(i+1)*ns,:,0].T
        x_hist[N:2*N,:] = x_test[i*ns:(i+1)*ns,:,1].T
        x_hist[2*N:3*N,:] = x_test[i*ns:(i+1)*ns,:,2].T
        x_hist[3*N:,:] = x_test[i*ns:(i+1)*ns,:,3].T
        
        x_pred = torch.zeros((4*N,ns))
        x_pred[:N,:] = x_approx[i*ns:(i+1)*ns,:,0].T
        x_pred[N:2*N,:] = x_approx[i*ns:(i+1)*ns,:,1].T
        x_pred[2*N:3*N,:] = x_approx[i*ns:(i+1)*ns,:,2].T
        x_pred[3*N:,:] = x_approx[i*ns:(i+1)*ns,:,3].T
        
        
        full_rel_error = np.linalg.norm(x_pred-x_hist,'fro')/np.linalg.norm(x_hist,'fro')
        errors[i] = np.linalg.norm(x_pred-x_hist,'fro')/np.linalg.norm(x_hist,'fro')
        
        print(full_rel_error)
    
    
    # np.save('Cyl_solutions/Cyl_GNN_recon_'+str(lat_dim),errors)


