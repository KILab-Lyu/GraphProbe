import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from functools import partial
from torch_geometric.data.feature_store import TensorAttr
import networkx as nx
import igraph
from sklearn.metrics import precision_score, recall_score, f1_score
from Utility.parser import args
from Utility.constant import *
from Datasets import select_data


class CentProbe(nn.Module):
    def __init__(self, emb_dim, hidden_dim=8, hidden_layers_num=5):
        super().__init__()
        self.linear_first = nn.Linear(emb_dim * 2, hidden_dim)
        self.linear_hid = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers_num)])
        self.linear_final = nn.Linear(hidden_dim, 3)

    def forward(self, emb1, emb2):
        h = torch.cat([emb1, emb2], dim=-1)
        h = self.linear_first(h)
        if len(self.linear_hid) != 0:
            for linear in self.linear_hid:
                h = linear(h)
        out = self.linear_final(h)
        return out


def cal_centrality_label(centrality, node1, node2):
    if centrality[node1] > centrality[node2]:
        return 0
    elif centrality[node1] == centrality[node2]:
        return 1
    else:
        return 2


class ALLCentrality:
    def __init__(self):
        self.cent_store = {}

    def get(self, data_name):
        if self.cent_store.get(data_name, None) is None:
            dataset = select_data(data_name, args.meta_emb)
            Centrality = {
                # 'degreeC': [],
                'betweenC': [],
                # 'closeC': [],
                'eigenC': [],
            }
            node_list = []
            for i, data in enumerate(dataset):
                num_nodes = data.num_nodes
                rand = np.random.permutation(num_nodes)
                nodelist1 = np.arange(num_nodes)
                nodelist2 = nodelist1[rand]
                node_list.append((nodelist1, nodelist2))

                networkxG = nx.Graph()
                networkxG.add_nodes_from(np.arange(num_nodes))
                networkxG.add_edges_from(data.edge_index.t().tolist())
                ig_G = igraph.Graph.from_networkx(networkxG)

                # degreeC = nx.degree_centrality(networkxG)
                # print(data_name, i, 'degreeC done!', end=' ')
                # betweenC = nx.betweenness_centrality(networkxG)
                # print('betweenC done!', end=' ')
                # closeC = nx.closeness_centrality(networkxG)
                # print('closeC done!', end=' ')
                # eigenC = nx.eigenvector_centrality(networkxG, max_iter=20000)
                # print('eigenC done!')

                # degreeC = ig_G.degree(nodelist1)
                # print(data_name, i, 'degreeC done!', end=' ')
                betweenC = ig_G.betweenness(nodelist1)
                # print('betweenC done!', end=' ')
                # closeC = ig_G.closeness(nodelist1)
                # print('closeC done!', end=' ')

                eigenC = ig_G.eigenvector_centrality(scale=True)

                # extract the centrality of specific nodes
                # ensure the node index in nodelist1 is in the range of the graph
                eigenC_temp = [eigenC[node] for node in nodelist1]
                eigenC = eigenC_temp
                # print('eigenC done!')

                # func_degree = partial(cal_centrality_label, degreeC)
                func_between = partial(cal_centrality_label, betweenC)
                # func_close = partial(cal_centrality_label, closeC)
                func_eigen = partial(cal_centrality_label, eigenC)

                # label_degree = list(map(func_degree, nodelist1, nodelist2))
                label_between = list(map(func_between, nodelist1, nodelist2))
                # label_close = list(map(func_close, nodelist1, nodelist2))
                label_eigen = list(map(func_eigen, nodelist1, nodelist2))

                # Centrality['degreeC'].append(torch.tensor(label_degree))
                Centrality['betweenC'].append(torch.tensor(label_between))
                # Centrality['closeC'].append(torch.tensor(label_close))
                Centrality['eigenC'].append(torch.tensor(label_eigen))


            self.cent_store[data_name] = Centrality, node_list
            print("{} betweenC is {}, eigenC is {}, max is {}, min is {}".format(data_name, sum(Centrality['betweenC'][0])/len(Centrality['betweenC'][0]),
                                                       sum(Centrality['eigenC'][0])/len(Centrality['eigenC'][0]), max(Centrality['betweenC'][0]), max(Centrality['eigenC'][0]
                                                       )))
        return self.cent_store[data_name]
    def calculate_homophily(graph, labels):
        homophily_ratios = []
        for node in graph.nodes():
            neighbors = graph.neighbors(node)
            if len(neighbors) == 0:
                continue
            same_label_count = sum(labels[neighbor] == labels[node] for neighbor in neighbors)
            homophily_ratio = same_label_count / len(neighbors)
            homophily_ratios.append(homophily_ratio)
        overall_homophily = sum(homophily_ratios) / len(homophily_ratios)
        return overall_homophily

all_centrality = ALLCentrality()


def train_CentProbe(emb, dataset):
    x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
    assert dataset.data.update_tensor(emb, x_attr)
    # if args.ge_model == "GCL":
    #     emb_dim = emb[0].shape[1]
    # else:
    emb_dim = emb.shape[1]

    Centrality, node_list = all_centrality.get(dataset.name)
    setattr(dataset, 'is_train', True)

    loss_fn = torch.nn.CrossEntropyLoss()

    f1_cent = {}
    for cent in Centrality.keys():
        centrality = Centrality[cent]

        probe = CentProbe(emb_dim=emb_dim, hidden_dim=emb_dim).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=0.005)

        for epoch in range(args.epochs):
            if epoch == args.epochs-1:
                all_pred = []
                all_label = []
            for i in range(len(dataset)):
                data = dataset[i]
                nodelist1, nodelist2 = node_list[i]
                label = centrality[i].long().to(device)
                node1 = data.x[nodelist1]
                node2 = data.x[nodelist2]
                pred = probe(node1.to(device), node2.to(device))
                loss = loss_fn(pred, label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if epoch == args.epochs - 1:
                    all_label.extend(label.cpu().tolist())
                    pred = torch.argmax(pred, dim=-1)
                    all_pred.extend(pred.cpu().tolist())

        f1_cent[cent] = f1_score(all_label, all_pred, average='micro')   # loss.item()

    return f1_cent


