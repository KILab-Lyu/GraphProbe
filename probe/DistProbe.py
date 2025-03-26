import torch
import torch.nn as nn
import torch.optim
import networkx as nx
from torch_geometric.data.feature_store import TensorAttr
from Utility.parser import args
from Utility.constant import *
from Datasets import select_data


class DistProbe(nn.Module):
    def __init__(self, emb_dim, rank=64, dist_mode='Euclidean', device='cuda'):
        super().__init__()
        self.rank = rank
        self.emb_dim = emb_dim
        self.proj = torch.nn.Parameter(torch.zeros(self.emb_dim, self.rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)
        self.dist_mode = dist_mode

    def forward(self, emb1, emb2):
        diffs = emb1 - emb2
        squared_diffs = diffs.abs()  # pow(2)
        transform = torch.matmul(self.proj, self.proj.t())
        out = torch.matmul(squared_diffs, transform)
        out = torch.sum(out, dim=-1)
        return out

    def forward1(self, node_emb, node_pair):
        if self.dist_mode == 'Euclidean':
            pred_dist = self.cal_Euclidean_dist(node_emb, node_pair)
        elif self.dist_mode == 'Cosine':
            pred_dist = self.cal_Cosine_dist(node_emb)
        return pred_dist

    def loss_fn(self, pred, label):
        return torch.mean(torch.abs(pred - label))

    def cal_Euclidean_dist(self, node_emb, node_pair):
        transformed = torch.matmul(node_emb, self.proj)
        node_num, rank = transformed.size()
        transformed = transformed.unsqueeze(1)
        transformed = transformed.expand(-1, node_num, -1)
        transposed = transformed.transpose(0, 1)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        squared_distances = squared_distances[node_pair[:, 0], node_pair[:, 1]]
        return squared_distances ** 0.5

    def cal_Cosine_dist(self, node_emb):
        transformed = torch.matmul(node_emb, self.proj)
        node_num, rank = transformed.size()
        return node_num


class ALLDistance:
    def __init__(self,cutoff):
        self.path_len_store = {}
        self.cutoff = cutoff

    def get(self, data_name):
        if self.path_len_store.get(data_name, None) is None:
            dataset = select_data(data_name)
            setattr(dataset, 'is_train', True)
            node_pairs_list = []
            node_pairs_len_list = []
            for data in dataset:
                networkxG = nx.Graph(data.edge_index.t().tolist())
                all_pairs_shortest_path_length = dict(nx.all_pairs_shortest_path_length(networkxG, cutoff=self.cutoff))
                node_pairs_len = []
                node_pairs = []
                for source in all_pairs_shortest_path_length.keys():
                    for target in all_pairs_shortest_path_length[source].keys():
                        length = all_pairs_shortest_path_length[source][target]
                        if length != 0:
                            node_pairs_len.append(length)
                            node_pairs.append([source, target])

                rand = torch.randperm(len(node_pairs_len))[:65535]
                node_pairs = torch.tensor(node_pairs)[rand]
                node_pairs_len = torch.tensor(node_pairs_len, dtype=torch.float)[rand]
                node_pairs_list.append(node_pairs)
                node_pairs_len_list.append(node_pairs_len)

            self.path_len_store[data_name] = node_pairs_list, node_pairs_len_list
        return self.path_len_store[data_name]




def train_DistProbe(emb, dataset, cutoff):
    assert dataset.data.num_nodes == emb.shape[0]
    print("---------now cutoff is {}-----------".format(cutoff))
    all_distance = ALLDistance(cutoff=cutoff)
    # dataset.data.x = emb
    x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
    assert dataset.data.update_tensor(emb.detach(), x_attr)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    node_pairs_list, node_pairs_len_list = all_distance.get(dataset.name)
    probe = DistProbe(emb_dim=emb.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=0.005)
    probe.train()
    for epoch in range(args.epochs):
        loss = 0
        for i, data in enumerate(dataset):
            node_pairs = node_pairs_list[i]
            node_pairs_len = node_pairs_len_list[i]
            node1 = data.x[node_pairs[:, 0]]
            node2 = data.x[node_pairs[:, 1]]
            pred = probe(node1.to(device), node2.to(device))
            label = node_pairs_len.to(device)
            batch_loss = loss_fn(pred, label**2)

            batch_loss.backward()
            opt.step()
            opt.zero_grad()

            torch.cuda.empty_cache()
            loss += batch_loss.item()

        loss = loss / len(dataset)

    return loss**0.5
