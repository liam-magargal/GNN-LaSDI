import numpy as np
import sys
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy

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
    

def getGDLSPGsolutionImp(x_hat_hist,model,N_lat,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, labels1, labels2, labels3, labels4):
    
    cuda = torch.device('cuda')
    x_hat_hist = x_hat_hist.to(cuda)
    model = model.to(cuda)
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
    
    
    for t in range(Nt-1):
        # print(t)
        x_hat_curr = x_hat_hist[:,t]
        x_curr = (model.decode(x_hat_curr.reshape((1,N_lat)), 1, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach()
        phi_curr = x_curr[0,:,0]
        u_curr = x_curr[0,:,1]

        x_hat_next = x_hat_hist[:,t]
        
        while True:
            x_next = (model.decode(x_hat_next.reshape((1,N_lat)), 1, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21)).detach()
            phi_next = x_next[0,:,0]
            u_next = x_next[0,:,1]
            
            
            r = torch.zeros((2*N),device=cuda)
            J = torch.zeros((2*N, 2*N),device=cuda)
            
            m = alpha/np.pi*torch.arctan(gamma*(Delta - u_next))
            
            
            r[intInd] = phi_next[intInd] - phi_curr[intInd] - dt/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2))
            r[boundInd] = phi_next[boundInd] 
            
            J[intInd,intInd] = 1 - dt/tau*(-4*eps*eps/dx/dx - 1/2 + 3*phi_next[intInd] -3*phi_next[intInd]**2 + m[intInd]*(1 - 2*phi_next[intInd]))
            J[intInd,intInd-1] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,intInd+1] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,intInd-Nx] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,intInd+Nx] = -dt/tau*(eps*eps/dx/dx)
            J[intInd,N+intInd] = - dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))
            J[boundInd,boundInd] = 1
            
            
            r[N+intInd] = u_next[intInd] - u_curr[intInd] - dt*(1/dx/dx*(u_next[intInd-1] - 2*u_next[intInd] + u_next[intInd+1]) + 1/dx/dx*(u_next[intInd-Nx] - 2*u_next[intInd] + u_next[intInd+Nx]) + K/tau*(eps*eps/dx/dx*(phi_next[intInd-1] - 2*phi_next[intInd] + phi_next[intInd+1]) + eps*eps/dx/dx*(phi_next[intInd-Nx] - 2*phi_next[intInd] + phi_next[intInd+Nx]) - 1/2*phi_next[intInd] + 3/2*phi_next[intInd]**2 - phi_next[intInd]**3 + m[intInd]*(phi_next[intInd] - phi_next[intInd]**2)))
            J[N+intInd,N+intInd] = 1 - dt*(1/dx/dx*(-4)) - dt*(K/tau*(-gamma*alpha/np.pi)*(1/(1 + gamma*gamma*(Delta-u_next[intInd])**2))*(phi_next[intInd]-phi_next[intInd]**2))
            J[N+intInd,N+intInd-1] = - dt*(1/dx/dx)
            J[N+intInd,N+intInd+1] = - dt*(1/dx/dx)
            J[N+intInd,N+intInd-Nx] = - dt*(1/dx/dx)
            J[N+intInd,N+intInd+Nx] = - dt*(1/dx/dx)
            
            J[N+intInd,intInd] = -K*dt/tau*(-4*eps*eps/dx/dx - 1/2 + 3*phi_next[intInd] -3*phi_next[intInd]**2 + m[intInd]*(1 - 2*phi_next[intInd]))
            J[N+intInd,intInd-1] = -K*dt/tau*(eps*eps/dx/dx)
            J[N+intInd,intInd+1] = -K*dt/tau*(eps*eps/dx/dx)
            J[N+intInd,intInd-Nx] = -K*dt/tau*(eps*eps/dx/dx)
            J[N+intInd,intInd+Nx] = -K*dt/tau*(eps*eps/dx/dx)
            
            r[N+boundInd] = u_next[boundInd]
            J[N+boundInd,N+boundInd] = 1
        
        
            J_dec = torch.zeros((2*N,N_lat),device=cuda)
            J_decoder = (torch.func.jacfwd(model.decode, argnums=0)(x_hat_next.reshape((1,N_lat)),1,e1,e2,e3,e4,e5,u1,u2,u3,u4,pos1,pos2,pos3,pos4,pos5,ai_54,ai_43,ai_32,ai_21)).detach()
            J_dec[:N,:] = J_decoder[0,:,0,0,:]
            J_dec[N:,:] = J_decoder[0,:,1,0,:]
            
            
            psi = J@J_dec
            RHS = psi.T@r
            LHS = psi.T@psi
            
            
            if torch.linalg.norm(RHS)<tol:
                x_hat_hist[:,t+1] = x_hat_next
                break
            else:
                x_hat_next = x_hat_next - torch.linalg.solve(LHS,RHS)
            
            
    return x_hat_hist


if __name__ == "__main__":
    N_lat = int(sys.argv[1])

    a_all = [.3875, .4625, .5375, .6125]
    b_all = [.3875, .4625, .5375, .6125]

    alpha = .9
    gamma = 10
    K = 1
    tau = .0003
    eps = .01
    Delta = 1.

    testSol = 5

    latDimToCanny = [3, 5, 7]
    testSolsToCanny = [0, 3, 12, 15]
    
    dt = .00002
    Nt = 501
    
    tol = 1e-4
    
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
    x_hist_in = x_hist_in[:,:,:Nt]
    nSol = np.shape(x_hist_in)[0]
    
    x_hist = x_hist_in[testSol,:,:]
    
    
    cuda = torch.device('cuda')
    
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


    model = torch.load('Kobayashi_May5_full_' + str(N_lat))
    model = model.to(cuda)

    nRep = 10

    solCount = 0
    errors = np.zeros(16)
    totalTime = np.zeros(16)
    lowDimSolTime = np.zeros(16)
    decodeTime = np.zeros(16)
    encodeTime = np.zeros(16)
    for i in range(4):
        for j in range(4):
            a=a_all[i]
            b=b_all[j]
            for rep in range(nRep):
                ts = time.time()
                t1 = time.time()
                r0 = .075
                r = np.sqrt(np.square(coords[:,0]-a) + np.square(coords[:,1]-b))
                phi_init = .5*(1 - np.tanh((r-r0)/(2*eps*np.sqrt(2))))
                U_init = Delta*phi_init
                x_init = torch.zeros((1,N,2),device=cuda)
                x_init[0,:,0] = torch.tensor(phi_init,dtype=torch.float32).to(cuda)
                x_init[0,:,1] = torch.tensor(U_init,dtype=torch.float32).to(cuda)

                x_hat_hist = torch.zeros(N_lat,Nt,device=cuda)
                x_hat_hist[:,0] = model.encode(x_init, 1, e1, e2, e3, e4, e5, s1, s2, s3, s4,labels1,labels2,labels3,labels4).detach()
                t2 = time.time()
                encodeTime[solCount] += t2-t1

                t1 = time.time()
                x_hat_hist = getGDLSPGsolutionImp(x_hat_hist,model,N_lat,Nt,tau,eps,K,dx,dt,tol,N,intInd,boundInd, e1, e2, e3, e4, e5, s1, s2, s3, s4, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21, labels1, labels2, labels3, labels4)
                t2 = time.time()
                lowDimSolTime[solCount] += t2-t1
                t3 =time.time()
                x_approx = model.decode(x_hat_hist.T, Nt, e1, e2, e3, e4, e5, u1, u2, u3, u4, pos1, pos2, pos3, pos4, pos5, ai_54, ai_43, ai_32, ai_21).detach().cpu()
                t4 = time.time()
                decodeTime[solCount] += t4-t3
                te = time.time()
                totalTime[solCount] += te-ts
            if solCount in testSolsToCanny and N_lat in latDimToCanny:
                np.save('Kobayashi_solutions/GD_LSPG_'+str(N_lat) + '_testSol_'+str(solCount), x_hat_hist.detach().cpu().numpy())
            x_hist_approx = np.zeros((2*N,Nt))
            x_hist_approx[:N,:] = (x_approx[:,:,0]).T.cpu().detach().numpy()
            x_hist_approx[N:,:] = (x_approx[:,:,1]).T.cpu().detach().numpy()

            errors[solCount] = np.linalg.norm(x_hist_approx-x_hist_in[solCount,:,:],'fro')/np.linalg.norm(x_hist_in[solCount,:,:],'fro')
            print(np.linalg.norm(x_hist_approx-x_hist_in[solCount,:,:],'fro')/np.linalg.norm(x_hist_in[solCount,:,:],'fro'))

            solCount += 1
    
    np.save('Kobayashi_solutions/GDLSPG_'+str(N_lat)+'_errors_',errors)
    np.save('Kobayashi_solutions/GDLSPG_'+str(N_lat)+'_encodeTime',encodeTime/nRep)
    np.save('Kobayashi_solutions/GDLSPG_'+str(N_lat)+'_lowDimSolTime',lowDimSolTime/nRep)
    np.save('Kobayashi_solutions/GDLSPG_'+str(N_lat)+'_decodeTime',decodeTime/nRep)
    np.save('Kobayashi_solutions/GDLSPG_'+str(N_lat)+'_totalTime',totalTime/nRep)
