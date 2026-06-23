import torch
import torch.nn as nn
import torch_geometric

from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from typing import Callable, Optional, Tuple
from torch import Tensor

import sys
import time
import numpy as np


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
    

def trainModel(model, optimizer, w_train, w_val, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, labels1, labels2, labels3, labels4, N_lat):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    model = model.to(device)
    
    train_loss_hist = np.zeros((maxEpoch))
    val_loss_hist = np.zeros((maxEpoch))
    ns_train = w_train.size()[0]
    ns_val = w_val.size()[0]
    
    train_set = np.arange(ns_train)
    val_set = np.arange(ns_val)
    
    w_train = w_train.to(device)
    w_val = w_val.to(device)
    
    e1 = e1.to(device)
    e2 = e2.to(device)
    e3 = e3.to(device)
    e4 = e4.to(device)
    e5 = e5.to(device)
    
    s1 = s1.to(device)
    s2 = s2.to(device)
    s3 = s3.to(device)
    s4 = s4.to(device)
    
    u1 = u1.to(device)
    u2 = u2.to(device)
    u3 = u3.to(device)
    u4 = u4.to(device)
    
    pos1 = pos1.to(device)
    pos2 = pos2.to(device)
    pos3 = pos3.to(device)
    pos4 = pos4.to(device)
    pos5 = pos5.to(device)
    
    ai_54 = ai_54.to(device)
    ai_43 = ai_43.to(device)
    ai_32 = ai_32.to(device)
    ai_21 = ai_21.to(device)
    
    labels1 = labels1.to(device)
    labels2 = labels2.to(device)
    labels3 = labels3.to(device)
    labels4 = labels4.to(device)
    
    for epoch in range(maxEpoch):
        t1 = time.time()
        np.random.shuffle(train_set)
        for i in range(625):
            optimizer.zero_grad()
            train_set_r = train_set[i*batchSize:(i+1)*batchSize]
        
            np.random.shuffle(val_set)
            val_set_r = val_set[0:batchSize]
        
            ## train loss
            inputs = w_train[train_set_r,:,:]
            x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)
            x_pred = model.decode(x_lat,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
            x_target = w_train[train_set_r,:,:]
            loss = torch.linalg.norm(x_pred - x_target)
            loss = loss*loss
            
            loss.backward()
            optimizer.step()
            
        
        ## validation loss
        inputs = w_val[val_set_r,:,:]
        x_lat = model.encode(inputs,batchSize, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4)
        x_pred = model.decode(x_lat,batchSize, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)
        x_target = w_val[val_set_r,:,:]
        val_loss = torch.linalg.norm(x_pred - x_target)
        val_loss = val_loss*val_loss
        
        t2 = time.time()
        print('epoch: ', epoch, ' train loss: ', loss.cpu().detach().numpy(), ', validation loss: ', val_loss.cpu().detach().numpy(), 't2-t1: ', t2-t1)
        train_loss_hist[epoch] = loss.cpu().detach().clone().numpy()
        val_loss_hist[epoch] = val_loss.cpu().detach().clone().numpy()
        
        
        torch.save(model, 'Kobayashi_May5_full_' + str(N_lat))
        
    return model



def loadDataTotal(numVal, N):
    
    
    
    x_hist_in = torch.tensor(np.load('Kobayashi_x_hist_imp_May5_full.npy'),dtype=torch.float32)
    ns = x_hist_in.size(dim=0)
    Nt = x_hist_in.size(dim=2)
    
    x_hist = torch.zeros(ns*Nt,N,2)
    for i in range(ns):
        x_hist[i*Nt:(i+1)*Nt,:,0] = x_hist_in[i,:N,:].T
        x_hist[i*Nt:(i+1)*Nt,:,1] = x_hist_in[i,N:,:].T
    
    
    total_set = np.arange(ns*Nt)
    np.random.shuffle(total_set)
    
    train_set = total_set[numVal:]
    val_set = total_set[:]
    
    x_val = x_hist[val_set,:,:]
    x_train = x_hist[train_set,:,:]
    
    
    return x_train, x_val


def defineModel(N_lat):
    model = architecture(N_lat)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    
    return model, optimizer


if __name__=="__main__":
    
    N_lat = int(sys.argv[1])
    N = 25600
    ns = 1
    numVal = 25 #50
    batchSize = 20
    maxEpoch = 2500 #25000
    
    directory = '2D_pooling_unpooling_Kobayashi_March18/' 
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
    
    print('loading data') 
    x_train, x_val = loadDataTotal(numVal, N)
    print('done loading data')

    model, optimizer = defineModel(N_lat)
    
    print('training model')
    model = trainModel(model, optimizer, x_train, x_val, batchSize, maxEpoch, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, labels1, labels2, labels3, labels4, N_lat)
