import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GraphConv
from torch_geometric.nn import BatchNorm, GCNConv, SAGEConv, GATConv, LGConv, SGConv, ChebConv, TAGConv, SSGConv, \
    GINConv, VGAE, global_add_pool, GraphConv
from torch_geometric.nn import MLP as Mlp, GIN
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

dropout_p = 0.0

class Chebyshev(torch.nn.Module):  # https://github.com/mdeff/cnn_graph
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = ChebConv(in_channels, out_channels, K=16)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.relu(node_emb)
        node_emb = self.dropout(node_emb)

        return node_emb


class SSGCN(torch.nn.Module):  # https://github.com/allenhaozhu/SSGC
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = SSGConv(in_channels, out_channels, alpha=0.05, K=16)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.dropout(node_emb)
        return node_emb

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GraphConv(in_channels, hidden_channels))
        for _ in range(hid_layers - 1):
            self.layers.append(GraphConv(hidden_channels, hidden_channels))
        self.layers.append(GraphConv(hidden_channels, out_channels))
        self.linear = torch.nn.Linear(out_channels, out_channels) 

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr) 
        x = self.linear(x)
        return x


class GCN(torch.nn.Module):  # https://github.com/tkipf/pygcn
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.relu(node_emb)
        node_emb = self.dropout(node_emb)

        # node_emb = self.conv3(node_emb, edge_index)
        # node_emb = self.relu(node_emb)
        # node_emb = self.dropout(node_emb)

        node_emb = self.conv2(node_emb, edge_index)
        return node_emb

class GCNTest(torch.nn.Module):  # https://github.com/tkipf/pygcn
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.relu(node_emb)
        node_emb = self.dropout(node_emb)

        # node_emb = self.conv3(node_emb, edge_index)
        # node_emb = self.relu(node_emb)
        # node_emb = self.dropout(node_emb)

        # node_emb = self.conv2(node_emb, edge_index)
        return node_emb

class LightGCN(torch.nn.Module):  # https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py  https://blog.csdn.net/u013602059/article/details/107792470
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.conv1 = LGConv(normalize=True)
        self.conv2 = LGConv(normalize=True)
        self.conv3 = LGConv(normalize=True)
        self.conv4 = LGConv(normalize=True)

    def forward(self, node_emb, edge_index):
        node_emb1 = self.conv1(node_emb, edge_index)
        node_emb2 = self.conv2(node_emb1, edge_index)
        # node_emb3 = self.conv3(node_emb2, edge_index)
        # node_emb4 = self.conv3(node_emb2, edge_index)

        node_emb = (node_emb + node_emb1 + node_emb2) * 0.3
        node_emb = self.linear(node_emb)
        return node_emb


class GraphSAGE(torch.nn.Module):  # https://zhuanlan.zhihu.com/p/79637787
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean', normalize=True, project=True)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean', normalize=True, project=True)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, aggr='mean', normalize=True, project=True)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean', normalize=True, project=True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.relu(node_emb)
        node_emb = self.dropout(node_emb)

        # node_emb = self.conv3(node_emb, edge_index)
        # node_emb = self.relu(node_emb)
        # node_emb = self.dropout(node_emb)

        node_emb = self.conv2(node_emb, edge_index)
        return node_emb


class GAT(torch.nn.Module):  # https://github.com/Diego999/pyGAT/blob/master/models.py
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.5)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=0.5)
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.5)

        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.elu(node_emb)
        node_emb = self.dropout(node_emb)

        # node_emb = self.conv3(node_emb, edge_index)
        # node_emb = self.elu(node_emb)
        # node_emb = self.dropout(node_emb)

        node_emb = self.conv2(node_emb, edge_index)
        return node_emb


class GIN(torch.nn.Module):  # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        mlp1 = Mlp([in_channels, hidden_channels])
        mlp3 = Mlp([hidden_channels, hidden_channels])
        mlp4 = Mlp([hidden_channels, hidden_channels])
        mlp2 = Mlp([hidden_channels, out_channels], dropout=0.5)
        self.conv1 = GINConv(nn=mlp1, train_eps=False)
        self.conv2 = GINConv(nn=mlp2, train_eps=False)
        self.conv3 = GINConv(nn=mlp3, train_eps=False)
        self.conv4 = GINConv(nn=mlp4, train_eps=False)
        self.elu = torch.nn.ELU()

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)

        # node_emb = self.conv3(node_emb, edge_index)
        # node_emb = self.conv4(node_emb, edge_index)

        node_emb = self.conv2(node_emb, edge_index)
        return node_emb


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv4 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.elu = torch.nn.ELU()


    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb)
        node_emb = self.elu(node_emb)
        node_emb = self.dropout(node_emb)

        # node_emb = self.conv3(node_emb)
        # node_emb = self.elu(node_emb)
        # node_emb = self.dropout(node_emb)

        node_emb = self.conv2(node_emb)
        return node_emb


class SGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = SGConv(in_channels, hidden_channels)
        self.conv2 = SGConv(hidden_channels, out_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.batch_norm2 = BatchNorm(out_channels)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.batch_norm1(node_emb)
        node_emb = F.dropout(node_emb.relu(), p=dropout_p, training=self.training)

        node_emb = self.conv2(node_emb, edge_index)
        node_emb = self.batch_norm2(node_emb)
        node_emb = F.dropout(node_emb.relu(), p=dropout_p, training=self.training)
        return node_emb


class TAG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.batch_norm2 = BatchNorm(out_channels)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.batch_norm1(node_emb)
        node_emb = F.dropout(node_emb.relu(), p=dropout_p, training=self.training)

        node_emb = self.conv2(node_emb, edge_index)
        node_emb = self.batch_norm2(node_emb)
        node_emb = F.dropout(node_emb.relu(), p=dropout_p, training=self.training)
        return node_emb





class WGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2, dropout=0.5,
                 structural_weight='degree'):
        super(WGCN, self).__init__()
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.structural_weight = structural_weight

        self.layers = torch.nn.ModuleList()
        self.layers.append(GraphConv(in_channels, hidden_channels))
        for _ in range(hid_layers - 1):
            self.layers.append(GraphConv(hidden_channels, hidden_channels))

        self.layers.append(GraphConv(hidden_channels, out_channels))
        self.linear = Linear(out_channels, out_channels)

    def compute_structural_weights(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()

        if self.structural_weight == 'degree':
            weights = deg
        else:
            weights = torch.ones_like(deg)

        
        weights = weights.unsqueeze(1) 
        return weights

    def forward(self, x, edge_index, edge_attr=None):
        weights = self.compute_structural_weights(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)
            x = x * weights
            x = self.relu(x)
            x = self.dropout(x)

        x = self.linear(x)
        return x



class LightGCL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2,
                 dropout=0.5, edge_dropout_rate=0.2, feature_mask_rate=0.2):
        super(LightGCL, self).__init__()
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.edge_dropout_rate = edge_dropout_rate
        self.feature_mask_rate = feature_mask_rate

        self.encoder = torch.nn.ModuleList()
       
        self.encoder.append(GraphConv(in_channels, hidden_channels))
       
        for _ in range(hid_layers - 1):
            self.encoder.append(GraphConv(hidden_channels, hidden_channels))
      
        self.encoder.append(GraphConv(hidden_channels, out_channels))

        self.linear = Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index1 = edge_dropout(edge_index, self.edge_dropout_rate)
        x1 = feature_mask(x, self.feature_mask_rate)

        edge_index2 = edge_dropout(edge_index, self.edge_dropout_rate)
        x2 = feature_mask(x, self.feature_mask_rate)

        z1 = self.encode(x1, edge_index1)
        z2 = self.encode(x2, edge_index2)

        return z1, z2

    def encode(self, x, edge_index):
        for layer in self.encoder:
            x = layer(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        z = self.linear(x)
        return z

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


import torch
from torch.nn import ReLU, Dropout, Linear
from torch_geometric.nn import GraphConv, GCNConv
import torch.nn.functional as F


class GCL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2,
                 dropout=0.5, edge_dropout_rate=0.2, feature_mask_rate=0.2,
                 augmentations=['edge_perturbation', 'feature_masking']):
        super(GCL, self).__init__()
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.edge_dropout_rate = edge_dropout_rate
        self.feature_mask_rate = feature_mask_rate
        self.augmentations = augmentations

        self.encoder = torch.nn.ModuleList()
        self.encoder.append(GCNConv(in_channels, hidden_channels))
        for _ in range(hid_layers - 1):
            self.encoder.append(GCNConv(hidden_channels, hidden_channels))
        self.encoder.append(GCNConv(hidden_channels, out_channels))
        self.linear = Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        edge_index1 = edge_perturbation(edge_index, num_edges,
                                        perturb_prob=self.edge_dropout_rate) if 'edge_perturbation' in self.augmentations else edge_index
        x1 = feature_masking(x, mask_prob=self.feature_mask_rate) if 'feature_masking' in self.augmentations else x
        edge_index2 = edge_perturbation(edge_index, num_edges,
                                        perturb_prob=self.edge_dropout_rate) if 'edge_perturbation' in self.augmentations else edge_index
        x2 = feature_masking(x, mask_prob=self.feature_mask_rate) if 'feature_masking' in self.augmentations else x
        z1 = self.encode(x1, edge_index1)
        z2 = self.encode(x2, edge_index2)

        return z1, z2

    def encode(self, x, edge_index):
        for layer in self.encoder:
            x = layer(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        z = self.linear(x)
        return z


from torch_geometric.utils import dropout_edge, dropout_node


def node_dropping(edge_index, num_nodes, drop_prob=0.2):
    mask = torch.bernoulli((1 - drop_prob) * torch.ones(num_nodes)).bool()
    nodes = torch.arange(num_nodes)
    kept_nodes = nodes[mask]
    mapping = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    mapping[kept_nodes] = torch.arange(kept_nodes.size(0), device=edge_index.device)
    edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
    edge_index = mapping[edge_index]
    return edge_index, mask


def edge_perturbation(edge_index, num_edges, perturb_prob=0.2):
    edge_index = dropout_edge(edge_index, p=perturb_prob)[0]
    num_new_edges = int(num_edges * perturb_prob)
    src = torch.randint(0, edge_index.max().item() + 1, (num_new_edges,), device=edge_index.device)
    dst = torch.randint(0, edge_index.max().item() + 1, (num_new_edges,), device=edge_index.device)
    new_edges = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, new_edges], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def feature_masking(x, mask_prob=0.2):
    mask = torch.bernoulli((1 - mask_prob) * torch.ones_like(x)).bool()
    return x * mask

def augment(graph):
    edge_drop = EdgePerturbation(p=0.2)
    node_drop = torch_geometric.transforms.RandomNodeDropout(p=0.2)
    attr_jitter = torch_geometric.transforms.RandomNodeAttrJitter(p=0.2)
    graph = edge_drop(graph)
    graph = node_drop(graph)
    graph = attr_jitter(graph)
    return graph

class GraphRNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.hid_layers = hid_layers

        # Assuming GIN is a defined Graph Neural Network layer elsewhere in your code
        self.rnn = GIN(in_channels, hidden_channels,out_channels)

        # Changed the input dimension to match the output of rnn layer
        self.mlp = nn.Linear(out_channels, out_channels)

    def forward(self, node_emb, edge_index):

        for i in range(self.hid_layers):
            h = self.rnn(node_emb, edge_index)
            node_emb = h

            # Uncommented this line to ensure the mlp layer is used in the forward pass
        node_emb = self.mlp(node_emb)

        return node_emb


class SSL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs_gin = nn.ModuleList()
        self.convs_gcn = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = MLP(input_dim, hidden_dim,output_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim,output_dim)
            conv_gin = GIN(input_dim, hidden_dim, output_dim)
            conv_gcn = GCN(input_dim, hidden_dim, output_dim)
            self.convs_gin.append(conv_gin)
            self.convs_gcn.append(conv_gcn)
            self.mlps.append(mlp)
        self.linear = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, node_emb, edge_index):
        x_gin = node_emb
        x_gcn = node_emb
        for i in range(self.num_layers):
            x_gin = self.convs_gin[i]
            x_gin = x_gin(node_emb,edge_index)
            x_gin = self.dropout(x_gin)
            x_gcn = self.convs_gcn[i]
            x_gcn = x_gcn(node_emb,edge_index)
            x_gcn = self.dropout(x_gcn)
        x_gin = self.linear(x_gin)
        x_gcn = self.linear(x_gcn)
        x = self.fc(x_gin + x_gcn)
        return x



class Node2VecModel(torch.nn.Module):
    def __init__(self, edge_index, embedding_dim=128, walk_length=20, context_size=10,
                 walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True):

        super(Node2VecModel, self).__init__()
        self.node2vec = Node2Vec(
            edge_index=to_undirected(edge_index),
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=sparse
        )
        self.embedding = self.node2vec.embedding

    def forward(self, x=None, edge_index=None):
        return self.node2vec.embedding.weight


import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

class DeepWalkModel(torch.nn.Module):
    def __init__(self, edge_index, num_nodes, embedding_dim=128, walk_length=40,
                 context_size=10, walks_per_node=10, num_negative_samples=1, sparse=True):

        super(DeepWalkModel, self).__init__()
        self.node2vec = Node2Vec(
            edge_index=to_undirected(edge_index),
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=1,
            q=1,
            sparse=sparse
        )

        self.embedding = self.node2vec.embedding

    def forward(self, x=None, edge_index=None):
        return self.node2vec.embedding.weight
