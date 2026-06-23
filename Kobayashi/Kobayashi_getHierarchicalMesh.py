import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import radius_graph
from sklearn.cluster import KMeans



def segment(N, Nc, edge_index, pos):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # assemble adjacency matrix and graph Laplacian
    A = torch_geometric.utils.to_dense_adj(edge_index).reshape((N,N))
    D = torch.diag(torch.sum(A,dim=1))
    L = D - A
    L = L.to(device)
    
    # compute eigendecomposition
    print('starting eigendecomp')
    l, v = torch.linalg.eig(L)
    print('done eigendecomp')
    l = l.cpu()
    v = v.cpu()
    
    l_sorted, indices = torch.sort(torch.abs(l),descending=False)
    
    U = torch.real(v[:,indices[1:(Nc)]])
    
    # perform k-means clustering on spectral features
    print('starting kmeans')
    model_kmeans = KMeans(n_clusters=Nc,n_init=Nc)
    model_kmeans = model_kmeans.fit(U)
    labels = model_kmeans.predict(U)
    labels = torch.tensor(labels,dtype=torch.int64)
    print('done kmeans')
    
    # generate assignment matrix
    S = torch.zeros((1,N,Nc))
    for i in range(N):
        j = labels[i]
        S[0,i,j] = 1.
    
    summation = torch.sum(S,dim=1).reshape((1,1,Nc))
    unpool = S.transpose(1,2).clone()
    S = torch.div(S,summation)
    
    ## rescale positions
    print(S.dtype, pos.dtype)
    pos2 = torch.matmul(S.transpose(1,2), pos)[0,:,:]
    rBound = torch.max(pos2[:,0])
    lBound = torch.min(pos2[:,0])
    pos2[:,0] = (pos2[:,0] - lBound*torch.ones((Nc))) / (rBound - lBound) * (torch.max(pos[:,0]) - torch.min(pos[:,0])) + torch.min(pos[:,0])
    
    tBound = torch.max(pos2[:,1])
    bBound = torch.min(pos2[:,1])
    pos2[:,1] = (pos2[:,1] - bBound*torch.ones((Nc))) / (tBound - bBound) * (torch.max(pos[:,1]) - torch.min(pos[:,1])) + torch.min(pos[:,1])
    
    ## get edge_index of coarsened graph
    edge_index2 = radius_graph(x=pos2,r=torch.sqrt(torch.tensor(9.*((torch.max(pos[:,0]) - torch.min(pos[:,0]))*(torch.max(pos[:,1]) - torch.min(pos[:,1])))/torch.pi/Nc)),loop=False)

    
    return S, unpool, edge_index2, pos2, labels



nStep = 160
x = np.linspace(0,1,nStep)
y = np.linspace(0,1,nStep)
dx = x[1] - x[0]


Nx = np.shape(x)[0]
Ny = Nx
N = Nx*Ny

X, Y = np.meshgrid(x, y)
pos = np.vstack([X.ravel(), Y.ravel()]).T
pos = torch.tensor(pos,dtype=torch.float32)


n1 = pos.size()[0]
n2 = 512
n3 = 64
n4 = 8
n5 = 2


edge_index = radius_graph(x=pos,r=torch.sqrt(torch.tensor(9.*((torch.max(pos[:,0]) - torch.min(pos[:,0]))*(torch.max(pos[:,1]) - torch.min(pos[:,1])))/torch.pi/n1)),loop=False)


# generate hierarchy of graphs
s1, unpool1, e2, pos2, labels1 = segment(n1,n2,edge_index,pos)
s2, unpool2, e3, pos3, labels2 = segment(n2,n3,e2,pos2)
s3, unpool3, e4, pos4, labels3 = segment(n3,n4,e3,pos3)
s4, unpool4, e5, pos5, labels4 = segment(n4,n5,e4,pos4)

# save outputs
directory = '2D_pooling_unpooling_Kobayashi_March18/'
torch.save(edge_index, directory + 'edge_index')
torch.save(e2, directory + 'e2')
torch.save(e3, directory + 'e3')
torch.save(e4, directory + 'e4')
torch.save(e5, directory + 'e5')

torch.save(s1, directory + 's1')
torch.save(s2, directory + 's2')
torch.save(s3, directory + 's3')
torch.save(s4, directory + 's4')

torch.save(unpool1, directory + 'unpool1')
torch.save(unpool2, directory + 'unpool2')
torch.save(unpool3, directory + 'unpool3')
torch.save(unpool4, directory + 'unpool4')

torch.save(pos, directory + 'pos1')
torch.save(pos2, directory + 'pos2')
torch.save(pos3, directory + 'pos3')
torch.save(pos4, directory + 'pos4')
torch.save(pos5, directory + 'pos5')

torch.save(labels1, directory + 'labels1')
torch.save(labels2, directory + 'labels2')
torch.save(labels3, directory + 'labels3')
torch.save(labels4, directory + 'labels4')
