import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,Linear
from utils import normalize_A, generate_cheby_adj

class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution( in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result

class DGCNN(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)
        return result

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = torch.randn(40, 64, 160).to(device)
    model = DGCNN(in_channels=160, num_electrodes=64, 
                  k_adj=2, out_channels=128, 
                  num_classes=105).to(device)
    
    clsout = model(tensor)
    print(clsout.shape)
