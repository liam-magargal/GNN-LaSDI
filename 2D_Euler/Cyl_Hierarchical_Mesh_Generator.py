import numpy as np
import pandas as pd
import time
import torch
import torch_geometric

from torch_kmeans import KMeans
from torch_geometric.nn import radius_graph

def segment(N, Nc, edge_index, pos):
    cuda = torch.device('cuda')
    print('starting segment')
    A = torch_geometric.utils.to_dense_adj(edge_index).reshape((N,N))
    D = torch.diag(torch.sum(A,dim=1))
    
    L = D - A
    L = L.to(cuda)
    
    print('starting eigenvalue decomp...')
    t1 = time.time()
    l, v = torch.linalg.eig(L)
    t2 = time.time()
    print('ending eigenvalue decomp...')
    print('eigendecomp time: ', t2-t1)
    l = l.cpu()
    v = v.cpu()
    
    l_sorted, indices = torch.sort(torch.abs(l),descending=False)
    
    U = torch.real(v[:,indices[1:(Nc)]])
    U = U.reshape((1,N,Nc-1))
    
    reduced_set = np.arange(N)
    U_reduced = U[:,reduced_set[:N],:]
    
    print('beginning kmeans')
    model_kmeans = KMeans(n_clusters=Nc)
    t1 = time.time()
    model_kmeans = model_kmeans.fit(U_reduced)
    t2 = time.time()
    print('kmeans time: ', t2-t1)

    labels = model_kmeans.predict(U)
    labels = labels.flatten().cpu()
    
    print('ending kmeans')

    S = torch.zeros((1,N,Nc))
    for i in range(N):
        j = labels[i]
        S[0,i,j] = 1.

    Ac = torch.matmul(A,S)
    Ac = torch.matmul(S.transpose(1,2),Ac)
    
    
    summation = torch.sum(S,dim=1).reshape((1,1,Nc))
    unpool = S.transpose(1,2).clone()
    S = torch.div(S,summation)
    
    ## normalize positions
    pos2 = torch.matmul(S.transpose(1,2), pos)[0,:,:]
    rBound = torch.max(pos2[:,0])
    lBound = torch.min(pos2[:,0])
    pos2[:,0] = (pos2[:,0] - lBound*torch.ones((Nc))) / (rBound - lBound) * .5
    
    tBound = torch.max(pos2[:,1])
    bBound = torch.min(pos2[:,1])
    pos2[:,1] = (pos2[:,1] - bBound*torch.ones((Nc))) / (tBound - bBound) * 1.
    
    ## get edge_index of clustered graph
    edge_index2 = radius_graph(x=pos2,r=torch.sqrt(torch.tensor(9./2./torch.pi/Nc)),loop=False)

    return S, unpool, edge_index2, pos2, labels

print('loading mesh')
pos = torch.tensor(np.load('Cyl_pooling_unpooling_normalized/cell_centroids.npy'),dtype=torch.float32)
print('done loading mesh')


n1 = pos.size()[0]
n2 = 512
n3 = 64
n4 = 8
n5 = 2


edge_index = radius_graph(x=pos,r=torch.sqrt(torch.tensor(9./2./torch.pi/n1)),loop=False)
t0 = time.time()
s1, unpool1, e2, pos2, labels1 = segment(n1,n2,edge_index,pos)
t1 = time.time()
s2, unpool2, e3, pos3, labels2 = segment(n2,n3,e2,pos2)
t2 = time.time()
s3, unpool3, e4, pos4, labels3 = segment(n3,n4,e3,pos3)
t3 = time.time()
s4, unpool4, e5, pos5, labels4 = segment(n4,n5,e4,pos4)
t4 = time.time()


assignments1 = torch.cat((pos,labels1.reshape(n1,1)),1)
assignments2 = torch.cat((pos2,labels2.reshape(n2,1)),1)
assignments3 = torch.cat((pos3,labels3.reshape(n3,1)),1)
assignments4 = torch.cat((pos4,labels4.reshape(n4,1)),1)

directory = 'Cyl_pooling_unpooling_normalized/'
torch.save(edge_index, directory + 'edge_index')
torch.save(e2, directory + 'e2')
torch.save(e3, directory + 'e3')
torch.save(e4, directory + 'e4')
torch.save(e5, directory + 'e5')

x1 = torch.rand((n1,1))
x2 = torch.rand((n2,1))
x3 = torch.rand((n3,1))
x4 = torch.rand((n4,1))
data = torch_geometric.data.Data(x=x2,edge_index=e2)
g = torch_geometric.utils.to_networkx(data, to_undirected=True)

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


df = pd.DataFrame(assignments1)
df.to_csv(directory+'assignments1.csv')
df = pd.DataFrame(assignments2)
df.to_csv(directory+'assignments2.csv')
df = pd.DataFrame(assignments3)
df.to_csv(directory+'assignments3.csv')
df = pd.DataFrame(assignments4)
df.to_csv(directory+'assignments4.csv')
