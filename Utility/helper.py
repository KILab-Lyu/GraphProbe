import os
from importlib import import_module
from importlib.util import find_spec

import dgl
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch_geometric.nn import HGTConv, KGEModel
from torch_geometric.nn.models import InnerProductDecoder, VGAE
# from torch_geometric.nn.kge import
from torch_geometric.datasets import FB15k_237

from GE_model import GraphRNN, SSL, GCNTest, GCNTest_agg, GNN, WGCN, LightGCL, DeepWalkModel, Node2VecModel, GCL

# from Utility.WGCN import utils_data
# from Utility.WGCN.utils_structural import load_adj, compute_structural_infot, generate_dijkstra
from Utility.constant import *
# from Utility.model.general_recommender.SGL import SGL, _LightGCN
# from Utility.reckit import Configurator
import Utility.model.GCL.Utility.losses as L
from Utility.model.GCL.Utility.models import get_sampler, DualBranchContrast
# from Utility.reckit import Configurator


class Stop_trick:
    def __init__(self, max_epoch=100, patience=50):
        self.max_epoch = max_epoch
        self.GE_Model = None
        self.DownStreamTask = None
        self.best = 10e5
        self.best_epoch = -1
        self.patience = patience
        self.patience_cnt = 0

    def should_stop_training(self, check_value, epoch, GE_Model, DownStreamTask):
        if check_value < self.best:
            self.GE_Model = GE_Model.state_dict()
            self.DownStreamTask = DownStreamTask.state_dict()
            self.best = check_value
            self.best_epoch = epoch
            self.patience_cnt = 0
        else:
            self.patience_cnt += 1

        should_stop = False
        if self.patience_cnt >= self.patience:
            should_stop = True
            reason = 'early stop'

        if epoch >= self.max_epoch-1:
            should_stop = True
            reason = 'max epoch'

        if should_stop:
            return should_stop, self.best_epoch, reason, [self.GE_Model, self.DownStreamTask]
        else:
            return False, None, None, [self.GE_Model, self.DownStreamTask]


def select_GE_model(ge_model, num_node_features, node_emb_dim, device, edge_index = None):
    from GE_model import GAT, GCN, GraphSAGE, MLP, LightGCN, SGCN, Chebyshev, TAG, GIN, SSGCN
    from torch_geometric.nn import Node2Vec
    from Utility.constant import List_GE_Model
    print(num_node_features)
    print(node_emb_dim)
    # if ge_model not in List_GE_Model: return
    if ge_model == 'GAT':
        GE_model = GAT(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim, heads=2)
    elif ge_model == 'GCN':
        GE_model = GCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GCNTest':
        GE_model = GCNTest(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GCNTest_agg':
        GE_model = GCNTest(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GNN':
        GE_model = GNN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GraphSAGE':
        GE_model = GraphSAGE(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'MLP':
        GE_model = MLP(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'LightGCN':
        GE_model = LightGCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'SGCN':
        GE_model = SGCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'Chebyshev':
        GE_model = Chebyshev(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'TAG':
        GE_model = TAG(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'SSGCN':
        GE_model = SSGCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GIN':
        GE_model = GIN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'WGCN':

        GE_model = WGCN(in_channels=num_node_features,hidden_channels=256,out_channels=node_emb_dim)

    elif ge_model == 'Node2Vec':#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
        GE_model = Node2VecModel(
            edge_index,
            embedding_dim=node_emb_dim,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        )
    elif ge_model == 'DeepWalk':
        GE_model = DeepWalkModel(
            edge_index=edge_index,
            num_nodes=edge_index.max().item()+1,
            embedding_dim=64,
            walk_length=30,
            context_size=10,
            walks_per_node=20,
            num_negative_samples=5,
            sparse=True
        )

    elif ge_model == 'VGAE':
        decoder = InnerProductDecoder()
        GE_model = VGAE(encoder=GCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim), decoder=decoder)

    elif ge_model == 'VGAE-GNN':
        decoder = InnerProductDecoder()
        GE_model = VGAE(encoder=GNN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim), decoder=decoder)

    elif ge_model == 'SGL':
        GE_model = SSL(input_dim=num_node_features, output_dim=node_emb_dim,hidden_dim=256)
    elif ge_model == "LightGCL":
        GE_model = LightGCL(input_dim=num_node_features,hidden_dim=256,output_dim=node_emb_dim)
    elif ge_model == "HGT":
        from torch_geometric.datasets import MovieLens
        dataset = MovieLens("../Data/MovieLens")
        data = dataset[0]
        metadata = data.metadata()
        GE_model = HGTConv(in_channels=num_node_features, out_channels=node_emb_dim, metadata=metadata)
    elif ge_model == "GraphRNN":
        GE_model = GraphRNN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == "Rotate":
        dataset = FB15k_237(root="../Data/FB15k_237")
        dataset.download()
        GE_model = KGEModel(
            num_relations=237,
            num_nodes = 14541,
            hidden_channels = 256
        )

    elif ge_model == "GCL":
        GE_model = GCL(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim).to(device)

    else:
        GE_model = GCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)

    return GE_model.to(device=device)


def select_GETest_model(ge_model, num_node_features, node_emb_dim, device, edge_index = None):
    from GETest_model import GAT, GCN, GraphSAGE, MLP, LightGCN, SGCN, Chebyshev, TAG, GIN, SSGCN
    print("Now is GE_model with GCN methods")
    from torch_geometric.nn import Node2Vec
    from Utility.constant import List_GE_Model
    # print(num_node_features)
    # print(node_emb_dim)
    # if ge_model not in List_GE_Model: return
    if ge_model == 'GAT'+"GCN":
        GE_model = GAT(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GCN'+"GCN":
        GE_model = GCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GCNTest'+"GCN":
        GE_model = GCNTest(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GCNTest_agg'+"GCN":
        GE_model = GCNTest(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GNN'+"GCN":
        GE_model = GNN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GraphSAGE'+"GCN":
        GE_model = GraphSAGE(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'MLP'+"GCN":
        GE_model = MLP(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'LightGCN'+"GCN":
        GE_model = LightGCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'SGCN':
        GE_model = SGCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'Chebyshev'+"GCN":
        GE_model = Chebyshev(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'TAG':
        GE_model = TAG(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'SSGCN'+"GCN":
        GE_model = SSGCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == 'GIN'+"GCN":
        GE_model = GIN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    # elif ge_model == 'WGCN':
    #
    #     origin_adj = load_adj("MovieLens")
    #     structural_info = compute_structural_infot("MovieLens", True, 1, 3, 0.0,0.4)
    #     #structural_info = compute_structural_info(args.dataset,origin_adj, args.directed, args.dijkstra_k, args.in_out_ratio,args.restart_rate,args.in_out_peak)
    #     structural_info = structural_info.toarray()
    #     generate_dijkstra("MovieLens", 1)
    #
    #     g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, num_devisions, pairs = utils_data.load_data(
    #         "MovieLens", 1, 4, None, 0.6, 0.2, 'WGCN', "poincare", structural_info,"bool","bool")
    #
    #
    #     g.set_n_initializer(dgl.init.zero_initializer)
    #     g.set_e_initializer(dgl.init.zero_initializer)
    #
    #     GE_model = WGCNNet(g=g, num_input_features=num_node_features, num_output_classes=18, num_hidden=256,
    #                        num_divisions=8, pairs = pairs, dropout_rate=0.5,
    #                        num_heads_layer_one=1, num_heads_layer_two=1,
    #                        layer_one_ggcn_merge='cat',
    #                        layer_one_channel_merge='cat',
    #                        layer_two_ggcn_merge='cat',
    #                        layer_two_channel_merge='mean', attention=False,layers=2)

    elif ge_model == 'Node2Vec'+"GCN":#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
        GE_model = Node2Vec(
            edge_index,
            embedding_dim=node_emb_dim,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        )
    elif ge_model == 'DeepWalk':
        from dgl.nn import DeepWalk
        from dgl.data import CoraGraphDataset
        dataset = CoraGraphDataset()
        GE_model = DeepWalk(dataset[0], emb_dim = node_emb_dim)

    elif ge_model == 'VGAE'+"GCN":
        decoder = InnerProductDecoder()
        GE_model = VGAE(encoder=GCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim), decoder=decoder)

    elif ge_model == 'VGAE-GNN':
        decoder = InnerProductDecoder()
        GE_model = VGAE(encoder=GNN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim), decoder=decoder)

    elif ge_model == 'SGL'+"GCN":
        GE_model = SSL(input_dim=num_node_features, output_dim=node_emb_dim,hidden_dim=256)
    elif ge_model == "LightGCL"+"GCN":
        GE_model = LightGCL(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == "HGT":
        from torch_geometric.datasets import MovieLens
        dataset = MovieLens("../Data/MovieLens")
        data = dataset[0]
        metadata = data.metadata()
        GE_model = HGTConv(in_channels=num_node_features, out_channels=node_emb_dim, metadata=metadata)
    elif ge_model == "GraphRNN":
        GE_model = GraphRNN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)
    elif ge_model == "Rotate":
        dataset = FB15k_237(root="../Data/FB15k_237")
        dataset.download()
        GE_model = KGEModel(
            num_relations=237,
            num_nodes = 14541,
            hidden_channels = 256
        )

    elif ge_model == "GCL":
        gconv = GConv(input_dim=num_node_features, hidden_dim=32, num_layers=2).to(device)

        # aug1 = A.Identity()
        # aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
        #                A.NodeDropping(pn=0.1),
        #                A.FeatureMasking(pf=0.1),
        #                A.EdgeRemoving(pe=0.1)], 1)
        #
        # class Encoder(torch.nn.Module):
        #     def __init__(self, encoder, augmentor):
        #         super(Encoder, self).__init__()
        #         self.encoder = encoder
        #         self.augmentor = augmentor
        #
        #     def forward(self, x, edge_index, batch):
        #         aug1, aug2 = self.augmentor
        #         x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        #         x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        #         z, g = self.encoder(x, edge_index, batch)
        #         z1, g1 = self.encoder(x1, edge_index1, batch)
        #         z2, g2 = self.encoder(x2, edge_index2, batch)
        #         return z, g, z1, z2, g1, g2
        #
        # encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
        GE_model = gconv

    else:
        GE_model = GCN(in_channels=num_node_features, hidden_channels=256, out_channels=node_emb_dim)

    return GE_model.to(device=device)


def gen_node_split_mask(node_num, batch_size, split_ratio=(0.6, 0.2, 0.2)):
    rand_order = torch.randperm(node_num)
    train_num, val_num = int(split_ratio[0] * node_num), int(split_ratio[1] * node_num)
    train_index = rand_order[:train_num]
    val_index = rand_order[train_num: train_num + val_num]
    test_index = rand_order[train_num + val_num:]

    def split_mask(data, batch_size):
        data_num = len(data)
        batch_num = data_num // batch_size + 1
        mask = torch.zeros((batch_num, data_num), dtype=torch.bool)
        i = 0
        while i < data_num:
            if i + batch_size < data_num:
                mask[i // batch_size][data[i:i + batch_size]] = True
            else:
                mask[i // batch_size][data[i:]] = True
            i += batch_size
        return mask

    train_mask = split_mask(train_index, batch_size)
    val_mask = split_mask(val_index, batch_size)
    test_mask = split_mask(test_index, batch_size)

    return train_mask, val_mask, test_mask


class Update_cal:
    def __init__(self, loss_fn=None):
        self.label_list = []
        self.pred_list = []
        self.loss_fn = loss_fn

    def update(self, label, pred):
        self.label_list.append(label)
        self.pred_list.append(pred)

    def cal(self, method):
        result = {}
        self.label_list = torch.cat(self.label_list, dim=0)
        self.pred_list = torch.cat(self.pred_list, dim=0)

        if 'loss' in method:
            result['loss'] = self.loss_fn(self.pred_list, self.label_list)

        self.label_list = self.label_list.detach().cpu()
        self.pred_list = self.pred_list.detach().cpu()
        label_np = self.label_list.detach().cpu().numpy()
        pred_np = self.pred_list.detach().cpu().numpy()

        if 'acc' in method:
            pred_list = torch.argmax(self.pred_list, dim=-1)
            result['acc'] = accuracy_score(self.label_list, pred_list)

        if 'f1' in method:
            result['f1'] = f1_score(self.label_list, (self.pred_list > 0.).float(), average='micro')

        if 'f1s' in method:
            pred_list = torch.argmax(self.pred_list, dim=-1)
            result['f1'] = f1_score(self.label_list, pred_list, average='micro')

        if 'auc' in method:
            result['auc'] = roc_auc_score(self.label_list, self.pred_list, average='macro')

        if 'f1c' in method:
            if self.pred_list.dim() > 1 and self.pred_list.size(1) > 1:
                # 多分类任务，取预测概率最大的类别
                pred_classes = torch.argmax(self.pred_list, dim=-1).detach().cpu().numpy()
                result['f1c'] = f1_score(label_np, pred_classes, average='macro')
            else:
                # 二分类任务，使用二分类的 F1 分数
                pred_classes = (self.pred_list > 0.5).astype(int).flatten()
                result['f1c'] = f1_score(label_np, pred_classes, average='binary')

        return result


dict_id2color={
    0:'g',
    1:'k',
    2:'b',
    3:'c',
    4:'m',
    5:'r',
    6:'y',
    7:'#9467bd',
}

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    points_color = list(map(lambda x:dict_id2color[x], points_color))
    # add_2d_scatter(ax, points, points_color, marker='o')
    # add_2d_scatter(ax, points, points_color, marker='s')

    x, y = points.T
    ax.scatter(x, y, c=points_color, s=20, alpha=0.8, marker='o')

    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'GraphSAGE-{title}.png', bbox_inches='tight')
    plt.show()


def add_2d_scatter(ax, points, points_color, marker):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=20, alpha=0.8, marker=marker)
    # ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())






"""
def minibatch(edges, num_user, item_list, batch_size):
    num_batch = num_user // batch_size+1
    user_list = torch.randperm(num_user)
    batch_user_list = []
    i = 0
    while i < num_user:
        right_bound = min(i + batch_size, num_user)
        batch = user_list[i:right_bound]
        batch_user_list.append(batch)
        i += batch_size

    assert len(batch_user_list) == num_batch

    batch_pos_edge_mask = torch.zeros((num_batch, len(edges)), dtype=torch.bool)
    for i, edge in enumerate(edges):
        found = False
        for j, batch_user in enumerate(batch_user_list):
            if edge[0] in batch_user:
                batch_pos_edge_mask[j, i] = True
                found = True
                break
        if found: continue

    batch_neg_edge_index = [[] for _ in range(num_batch)]
    batch_edge_num = [torch.where(pos_edge_mask)[0].shape[0] for pos_edge_mask in batch_pos_edge_mask]
    if item_list is not None:
        item_set = set(item_list.tolist())
        for i in range(num_batch):
            pos_item = set(edges[batch_pos_edge_mask[i]][:, 1].tolist())
            neg_item = list(item_set.difference(pos_item))
            batch_user = batch_user_list[i].tolist()
            batch_neg_edge_index[i] = [(choice(batch_user), choice(neg_item)) for _ in range(batch_edge_num[i])]

    return batch_pos_edge_mask, batch_neg_edge_index
"""