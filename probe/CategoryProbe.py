import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch_geometric.data.feature_store import TensorAttr
from sklearn.metrics import davies_bouldin_score
import sklearn.manifold as manifold
# from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding
from Utility.parser import args
from Utility.constant import *
from Utility.helper import plot_2d
from Datasets import select_data


class ALLCategory:
    def __init__(self):
        self.category_store = {}

    def get(self, data_name):
        if self.category_store.get(data_name, None) is None:
            dataset = select_data(data_name)
            # self.set_category = set(dataset.data.y.tolist())
            if data_name in ('cora', 'citeseer', 'pubmed', 'flickr'):
                node_category = []
                for i, data in enumerate(dataset):
                    node_category.append(data.y.tolist())

                # node_category = np.array(node_category)
                self.category_store[data_name] = node_category
            else:
                graph_category = []
                for i, data in enumerate(dataset):
                    graph_category.append(int(data.y))

                # graph_category = np.array(graph_category)

                self.category_store[data_name] = [graph_category]
        return self.category_store[data_name]


all_category = ALLCategory()


def category_probe(emb, dataset, method='ClusterProbe'):
    emb = emb.to(device='cpu')
    assert dataset.data.num_nodes == emb.shape[0]
    if dataset.name in ('mutag', 'proteins', 'enzymes'):
        slice = dataset.slices.get('x')
        entity_emb_list = torch.zeros((len(dataset), emb.shape[1]))
        for i in range(0, len(slice) - 1):
            node_index = list(range(slice[i], slice[i + 1]))
            entity_emb_list[i] = torch.sum(emb[node_index], dim=0)
        entity_emb_list = [entity_emb_list]

    elif dataset.name in ('cora', 'citeseer', 'pubmed', 'flickr'):
        x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
        assert dataset.data.update_tensor(emb, x_attr)
        entity_emb_list = []
        for data in dataset:
            entity_emb_list.append(data.x)

    category_list = all_category.get(dataset.name)

    method = 'TSNE'

    if method == 'ClusterProbe':
        sum = 0
        for entity_emb, category in zip(entity_emb_list, category_list):
            dbi = davies_bouldin_score(entity_emb.detach().numpy(), category)
            sum += dbi
        return sum / len(category_list)

    if method == 'ContrastiveProbe':
        sum = 0
        for entity_emb, category in zip(entity_emb_list, category_list):
            if type(category) is list:
                category = torch.tensor(category, dtype=torch.float)
            else:
                category = torch.from_numpy(category).float()
            if dataset.name in ('pubmed'):
                rand = torch.randperm(category.shape[0])[:5000]
                entity_emb1 = entity_emb[rand]
                category1 = category[rand]
                result = train_ContrastiveProbe(entity_emb1, category1)
            else:
                result = train_ContrastiveProbe(entity_emb, category)
            sum += result
        return sum / len(category_list)

    if method == 'TSNE':
        entity_emb_list = torch.cat(entity_emb_list, dim=0).numpy()
        category_list = category_list[0]
        X_embedded = manifold.TSNE(n_components=2, perplexity=10, learning_rate='auto', random_state=1, n_jobs=8).fit_transform(entity_emb_list)
        plot_2d(X_embedded, category_list, title='TSNE', id=None)



def train_ContrastiveProbe(emb0, label):
    probe = ContrastiveProbe(emb0.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    probe.train()
    for epoch in range(args.epochs):
        emb1 = probe(emb0)
        loss = contrastive_loss(emb1, label)
        loss.backward(retain_graph=True)
        opt.step()
        opt.zero_grad()

    dbi = davies_bouldin_score(emb1.detach().numpy(), label)
    return dbi


class ContrastiveProbe(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        self.linear3 = nn.Linear(emb_dim, emb_dim)
        nn.init.xavier_normal_(self.linear1.weight.data)
        nn.init.xavier_normal_(self.linear2.weight.data)
        nn.init.xavier_normal_(self.linear3.weight.data)

    def forward(self, emb):
        emb1 = self.linear1(emb)
        emb2 = self.linear2(emb1)
        emb3 = self.linear3(emb2)
        return emb3


def contrastive_loss(emb, label):
    t = torch.tensor(0.5)
    n = label.shape[0]
    similarity_matrix = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=2)
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
    mask_no_sim = torch.ones_like(mask) - mask
    mask_eye_0 = torch.ones(n, n) - torch.eye(n, n)

    similarity_matrix_t = torch.exp(similarity_matrix / t)

    similarity_matrix_t = similarity_matrix_t * mask_eye_0
    sim = mask * similarity_matrix_t
    no_sim = similarity_matrix_t - sim
    # no_sim = similarity_matrix_t * (torch.eye(n, n) - mask)

    no_sim_sum = torch.sum(no_sim, dim=1)

    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss1 = torch.div(sim, sim_sum)

    loss2 = mask_no_sim + loss1 + torch.eye(n, n)
    loss3 = -torch.log(loss2)
    loss = torch.sum(torch.sum(loss3, dim=1)) / (len(torch.nonzero(loss3)))

    return loss
