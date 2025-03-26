import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from sklearn.metrics import davies_bouldin_score
import random
from Utility.parser import args
from Utility.constant import *
from Utility.metrics import BPR_loss
from Datasets import select_data


class MIProbe(nn.Module):
    def __init__(self, emb_dim, hidden_dim=128, hidden_layers_num=3):
        super().__init__()
        self.linear_first = nn.Linear(emb_dim * 2, hidden_dim)
        self.linear_hid = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers_num)])
        self.linear_final = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear_first.weight.data)
        for l in self.linear_hid:
            nn.init.xavier_normal_(l.weight.data)
        nn.init.xavier_normal_(self.linear_final.weight.data)

    def forward(self, emb1, emb2):
        emb = torch.cat([emb1, emb2], dim=-1)
        h = self.linear_first(emb)
        if len(self.linear_hid) != 0:
            for l in self.linear_hid:
                h = l(h)
        out = self.linear_final(h)
        # out = F.elu(out)
        return out.squeeze(-1)


class ALLCategory:
    def __init__(self):
        self.category_store = {}

    def get(self, data_name):
        if self.category_store.get(data_name, None) is None:
            dataset = select_data(data_name)

            self.set_category = set(dataset.data.y.tolist())
            graph_emb_id = []
            graph_category = []
            for i, data in enumerate(dataset):
                graph_emb_id.append(i)
                graph_category.append(int(data.y))

            graph_emb_id = np.array(graph_emb_id)
            graph_category = np.array(graph_category)

            self.category_store[data_name] = (graph_emb_id, graph_category)
        return self.category_store[data_name]

    def gen_graph_pair(self, data_name, mode='same'):
        graph_emb_id, graph_category = self.get(data_name)
        set_category = set(graph_category.tolist())
        graph_id_list1 = []
        graph_id_list2 = []
        if mode == 'same':
            for c in set_category:
                graph1 = graph_emb_id[graph_category == c]
                rand_perm = np.random.permutation(graph1.size)
                graph2 = graph1[rand_perm]
                graph_id_list1 = np.append(graph_id_list1, graph1)
                graph_id_list2 = np.append(graph_id_list2, graph2)
        elif mode == 'other':
            for c in set_category:
                graph1 = graph_emb_id[graph_category == c]
                other_graph = graph_emb_id[graph_category != c]
                graph2 = random.choices(other_graph, k=graph1.shape[0])
                graph_id_list1 = np.append(graph_id_list1, graph1)
                graph_id_list2 = np.append(graph_id_list2, graph2)
        elif mode == 'self':
            for c in set_category:
                graph1 = graph_emb_id[graph_category == c]
                graph_id_list1 = np.append(graph_id_list1, graph1)
            graph_id_list2 = graph_id_list1.copy()
        elif mode == 'random':
            for c in set_category:
                graph1 = graph_emb_id[graph_category == c]
                graph_id_list1 = np.append(graph_id_list1, graph1)
            graph_id_list2 = np.random.random(graph_id_list1.size)
        elif mode == 'random_s':
            for c in set_category:
                graph1 = graph_emb_id[graph_category == c]
                graph_id_list1 = np.append(graph_id_list1, graph1)
            graph_id_list2 = np.random.random(graph_id_list1.size)

        graph_id_list1 = torch.from_numpy(graph_id_list1)
        graph_id_list2 = torch.from_numpy(graph_id_list2)
        return graph_id_list1.to(dtype=torch.long, device=device), graph_id_list2.to(dtype=torch.long, device=device)


all_category = ALLCategory()


def train_MIProbe(emb, dataset):
    def joint_distri(probe, graph_emb1, graph_emb2):
        joint = probe(graph_emb1, graph_emb2)
        joint = torch.mean(joint)
        return joint

    def marginal_distri(probe, graph_emb1, graph_emb2):
        marginal = probe(graph_emb1, graph_emb2)
        marginal = torch.log(torch.mean(torch.exp(torch.clamp(marginal, max=88))))
        return marginal

    assert dataset.data.num_nodes == emb.shape[0]
    dataset.data.x = emb
    graph_emb_list = torch.empty((len(dataset), dataset.data.x.shape[1]), device=device)
    for i, data in enumerate(dataset):
        graph_emb_list[i] = torch.mean(data.x, dim=0).to(dtype=torch.float, device=device)

    same1, same2 = all_category.gen_graph_pair(dataset.name, 'same')
    other1, other2 = all_category.gen_graph_pair(dataset.name, 'other')
    self1, self2 = all_category.gen_graph_pair(dataset.name, 'self')
    random1, _ = all_category.gen_graph_pair(dataset.name, 'random')
    random2 = torch.randn(graph_emb_list[random1].shape, dtype=torch.float, device=device)
    random3 = random2[torch.randperm(random2.shape[0])]
    probe = MIProbe(emb_dim=dataset.data.x.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        probe.train()
        joint = joint_distri(probe, graph_emb_list[same1], graph_emb_list[same2])
        marginal = marginal_distri(probe, graph_emb_list[other1], graph_emb_list[other1])
        train_mi = joint - marginal
        loss = -train_mi

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            probe.eval()


    with torch.no_grad():
        probe.eval()
        mi = joint_distri(probe, graph_emb_list[same1], graph_emb_list[same2]) - marginal_distri(probe, graph_emb_list[other1], graph_emb_list[other1])
        lower = joint_distri(probe, graph_emb_list[random1], random2) - marginal_distri(probe, graph_emb_list[random1], random3)
        upper = joint_distri(probe, graph_emb_list[self1], graph_emb_list[self2]) - marginal_distri(probe, graph_emb_list[same1], graph_emb_list[same2])
        probe_score = mi / (upper - lower)
        print(probe_score)
    return probe_score


def cal_ClusterProbe(emb, dataset):
    assert dataset.data.num_nodes == emb.shape[0]
    dataset.data.x = emb
    graph_emb_list = torch.empty((len(dataset), dataset.data.x.shape[1]))
    for i, data in enumerate(dataset):
        graph_emb_list[i] = torch.mean(data.x, dim=0)

    graph_emb_id, graph_category = all_category.get(dataset)
    dbi = davies_bouldin_score(graph_emb_list[graph_emb_id], graph_category)
    print(dbi)


