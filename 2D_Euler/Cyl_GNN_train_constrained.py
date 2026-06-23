import numpy as np
import sys
import os
import pandas as pd
import time
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



def trainModel(model, optimizer, w_train, w_val, w_test, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, N_lat):
    cuda = torch.device('cuda')
    model = model.to(cuda)
    
    N = w_train.size()[1]
    ns_train = w_train.size()[0]
    ns_val = w_val.size()[0]
    
    train_set = np.arange(ns_train)
    val_set = np.arange(ns_val)
    
    train_loss_hist = np.zeros((maxEpoch,1))
    test_loss_hist = np.zeros((maxEpoch,1))
    
    w_train = w_train.to(cuda)
    w_val = w_val.to(cuda)
    
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
    
    
    for epoch in range(maxEpoch):
        t1 = time.time()
        np.random.shuffle(train_set)
        for i in range(250):
            optimizer.zero_grad()
            train_set_r = train_set[i*batchSize:(i+1)*batchSize]
        
            np.random.shuffle(val_set)
            val_set_r = val_set[0:batchSize]
                
            ## train loss
        
            inputs = w_train[train_set_r,:,:]
            x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4)
            x_pred = torch.zeros((batchSize,N,4),device=cuda)
            x_pred = model.decode(x_lat,x_pred,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
            x_target = w_train[train_set_r,:,:]
            loss = torch.linalg.norm(x_pred - x_target)
            loss = loss*loss
            
            loss.backward()
            optimizer.step()
            
        ## validation loss
        inputs = w_val[val_set_r,:,:]
        x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4)
        x_pred = torch.zeros((batchSize,N,4),device=cuda)
        x_pred = model.decode(x_lat,x_pred,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
        x_target = w_val[val_set_r,:,:]
        val_loss = torch.linalg.norm(x_pred - x_target)
        val_loss = val_loss*val_loss
        
        
        t2 = time.time()
        print('epoch: ', epoch, ' train loss: ', loss.cpu().detach().numpy(), ', validation loss: ', val_loss.cpu().detach().numpy(), 't2-t1: ', t2-t1)
        
        train_loss_hist[epoch] = loss.detach().cpu().numpy()
        
        
        torch.save(model, 'Cyl_GNN_lat_'+str(N_lat)+'_500TS_constrained') 
        
        
    return model

def loadDataTotal(numTrain, numTest, numVal, ns, N):
    gamma = 1.4
    
    ni = 11
    x_hist = torch.zeros((ni*ns,N,4))
    x_hist_flat = torch.zeros((ni*ns*N,5))
    
    for i in range(ni):
        train_file = 'output'
        mesh_location = 'Cyl_pooling_unpooling_normalized/'
        output_location = 'Cyl_x_hist_500TS_'+str(1000+100*i)+'.npy'
        volume_file = 'cell_volumes.npy'
        x_hist_temp = torch.tensor(np.load(output_location))
        
        Mass = x_hist_temp[:N,:].T
        Mom_x = x_hist_temp[N:2*N,:].T
        Mom_y = x_hist_temp[2*N:3*N,:].T
        Energy = x_hist_temp[3*N:,:].T
        
        x_hist[i*ns:(i+1)*ns,:,0] = Mass
        x_hist[i*ns:(i+1)*ns,:,1] = Mom_x
        x_hist[i*ns:(i+1)*ns,:,2] = Mom_y
        x_hist[i*ns:(i+1)*ns,:,3] = Energy

    total_set = np.arange(ni*ns)
    np.random.shuffle(total_set)
    
    train_set = total_set[numVal:]
    val_set = total_set[:numVal]
    
    x_val = x_hist[val_set,:,:]
    x_train = x_hist[train_set,:,:]
    x_test = 0.
    
    
    return x_train, x_val, x_test


def defineModel(N_lat):
    model = architecture(N_lat)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) 
    
    milestones = []
    
    scheduler = None
    
    return model, optimizer, scheduler


def getAssignment(N,n2,n3,n4,n5):
    s1 = torch.zeros((1,N,n2))
    s2 = torch.zeros((1,n2,n3))
    s3 = torch.zeros((1,n3,n4))
    s4 = torch.zeros((1,n4,n5))
    
    for i in range(N):
        j = int(np.floor(i/(N/n2)))
        s1[0,i,j] = n2/N
    
    for i in range(n2):
        j = int(np.floor(i/(n2/n3)))
        s2[0,i,j] = n3/n2
        
    for i in range(n3):
        j = int(np.floor(i/(n3/n4)))
        s3[0,i,j] = n4/n3
        
    for i in range(n4):
        j = int(np.floor(i/(n4/n5)))
        s4[0,i,j] = n5/n4
    
    
    u1 = s1.transpose(1,2)*N/n2
    u2 = s2.transpose(1,2)*n2/n3
    u3 = s3.transpose(1,2)*n3/n4
    u4 = s4.transpose(1,2)*n4/n5
    
    
    return s1, s2, s3, s4, u1, u2, u3, u4


N = 4148
nt = 501
numTrain = 9
numTest = 8 # number of parameters used as test parameters
numVal = 511 #1 # number of snapshots from training stored as validation
batchSize = 20 
maxEpoch = 2500
N_lat = int(sys.argv[1])


directory = '../GD_LSPG/Cyl_pooling_unpooling_normalized/'
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

ai_54 = knn(pos5, pos4, k=3, batch_x=None, batch_y=None, num_workers=1)
ai_43 = knn(pos4, pos3, k=3, batch_x=None, batch_y=None, num_workers=1)
ai_32 = knn(pos3, pos2, k=3, batch_x=None, batch_y=None, num_workers=1)
ai_21 = knn(pos2, pos1, k=3, batch_x=None, batch_y=None, num_workers=1)


print('Loading data...')
w_train, w_val, w_test = loadDataTotal(numTrain, numTest, numVal, nt, N)
print(w_train.size())
print('maxes: ')
print(torch.max(w_train[:,:,0]))
print(torch.max(w_train[:,:,1]))
print(torch.max(w_train[:,:,2]))
print(torch.max(w_train[:,:,3]))
print('mins: ')
print(torch.min(w_train[:,:,0]))
print(torch.min(w_train[:,:,1]))
print(torch.min(w_train[:,:,2]))
print(torch.min(w_train[:,:,3]))
print((w_train.size()))

print('Done loading data...')


model, optimizer, scheduler = defineModel(N_lat)


model = trainModel(model, optimizer, w_train, w_val, w_test, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, N_lat)

