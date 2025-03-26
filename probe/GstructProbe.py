import numpy as np
import torch
import torch.nn as nn
import torch.optim
from scipy import stats
import pandas as pd
import networkx as nx
from Utility.parser import args
from Utility.constant import *
from Datasets import select_data
from torch_geometric.loader import DataLoader
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, WLConv
# import sklearn.manifold as manifold
from Utility.helper import plot_2d
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS, SpectralEmbedding


def cal_GstructProbe(emb, dataset):
    # cosine similarity
    assert dataset.data.num_nodes == emb.shape[0]
    slice = dataset.slices.get('x')
    graph_emb_list = torch.zeros((len(dataset), emb.shape[1]))
    for i in range(0, len(slice) - 1):
        node_index = list(range(slice[i], slice[i + 1]))
        graph_emb_list[i] = torch.mean(emb[node_index], dim=0)
    graph_emb_similarity = cal_graph_emb_similarity(graph_emb_list)

    method = None  #'TSNE'
    if method == 'TSNE':
        X_all = np.zeros([1,64])
        id_all = np.ones([1]) * 7
        # label_all =
        slice = dataset.slices.get('x')
        for i in range(0,8):
            node_index = list(range(slice[i], slice[i + 1]))
            graph_node_emb = emb[node_index]
            # X_embedded = TSNE(n_components=2, perplexity=10, learning_rate='auto', random_state=1, n_jobs=8).fit_transform(graph_node_emb)
            # X_embedded = LocallyLinearEmbedding(n_components=2, random_state=1, n_jobs=8).fit_transform(graph_node_emb)
            # X_embedded = Isomap(n_components=2, n_jobs=8).fit_transform(graph_node_emb)
            # X_embedded = MDS(n_components=2, random_state=1, n_jobs=8).fit_transform(graph_node_emb)
            # X_embedded = SpectralEmbedding(n_components=2, random_state=1, n_jobs=8).fit_transform(graph_node_emb)
            label = int(dataset[i].y)
            X_all = np.concatenate([X_all, graph_node_emb], axis=0)
            id_all = np.concatenate([id_all, i*np.ones(graph_node_emb.shape[0])], axis=0)

        X_all = TSNE(n_components=2, perplexity=10, learning_rate='auto', random_state=1, n_jobs=8).fit_transform(X_all)
        plot_2d(X_all, id_all, title='08')


    # WL similarity
    graph_node_set = {}
    wlconv = WLConv()
    for i, data in enumerate(dataset):
        ones = torch.ones(data.num_nodes, dtype=torch.long)
        out = wlconv(ones, data.edge_index)
        out = wlconv(out, data.edge_index)
        out = wlconv(out, data.edge_index)
        out = wlconv(out, data.edge_index)
        graph_node_set[i] = set(out.tolist())

    graph_WL_similarity = cal_graph_WL_similarity(graph_node_set)

    probe_score = cal_spearman_correlation_rank(graph_emb_similarity, graph_WL_similarity)
    # probe_score = cal_pearson_correlation(graph_emb_similarity, graph_WL_similarity)
    # probe_score = cal_mat_l2(graph_emb_similarity, graph_WL_similarity)
    return probe_score


def cal_mat_l2(mat1, mat2):
    mat1 = (mat1 - mat1.min()) / (mat1.max() - mat1.min())
    mat2 = (mat2 - mat2.min()) / (mat2.max() - mat2.min())
    probe_score = torch.mean((mat1 - mat2)**2)
    return probe_score.item()


def cal_pearson_correlation(mat1, mat2):
    corrcoef = np.corrcoef(mat1.view(-1), mat2.view(-1))
    probe_score = corrcoef[0, 1]
    return probe_score


def cal_spearman_correlation_rank(mat1, mat2):
    sort_index1 = torch.argsort(mat1, descending=True, dim=1)
    mat1 = convert_index2rank(sort_index1)

    sort_index2 = torch.argsort(mat2, descending=True, dim=1)
    mat2 = convert_index2rank(sort_index2)

    probe_score = stats.spearmanr(mat1.view(-1), mat2.view(-1))
    return probe_score.correlation


def cal_graph_emb_similarity(graph_emb, similarity_mode='cosine'):
    if similarity_mode == 'cosine':
        mat = torch.matmul(graph_emb, graph_emb.t())
        module = torch.norm(graph_emb, dim=1).unsqueeze(1)
        denom = torch.matmul(module, module.t())
        mat = mat / (denom ** 0.5)

    elif similarity_mode == 'euclid':
        expand_graph_emb = graph_emb.unsqueeze(dim=1)
        expand_graph_emb = expand_graph_emb.expand(-1, len(graph_emb), -1)
        mat = expand_graph_emb - expand_graph_emb.transpose(0, 1)
        mat = -torch.pow(mat, 2)
        mat = torch.sum(mat, dim=-1)

    return mat


def cal_graph_WL_similarity(graph_node_set):
    num_graph = len(graph_node_set)
    mat = torch.zeros((num_graph, num_graph))
    for graph_id1 in range(num_graph):
        for graph_id2 in range(num_graph):
            intersection = set.intersection(graph_node_set[graph_id1], graph_node_set[graph_id2])
            union = set.union(graph_node_set[graph_id1], graph_node_set[graph_id2])
            mat[graph_id1, graph_id2] = len(intersection) / len(union)

    return mat


def convert_index2rank(mat):
    #  torch.argsort 返回的结果为一个向量a，v = a[i]，排名为i的数来自于原数组的第v个数，先利用dict将key：value互换位置，i = r[v]，表示的是原数组第v个值的排名为i
    num_graph = len(mat)
    for i in range(num_graph):
        index2rank = {}
        rank = mat[i]
        for j in range(num_graph):
            index2rank[rank[j].item()] = j

        for j in range(num_graph):
            rank[j] = index2rank[j]

    return mat


# x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
# assert dataset.data.update_tensor(emb, x_attr)
# graph_emb_list = []
# graph_node_sets = []
#
# train_dataloader = DataLoader(dataset, batch_size=1)
# wlconv = WLConv()
# for data in train_dataloader:
#     graph_emb = torch.sum(data.x, dim=0)
#     graph_emb_list.append(graph_emb)
#
#     WL_emb1 = wlconv(torch.ones(data.num_nodes, dtype=torch.long), data.edge_index)
#     WL_emb2 = wlconv(WL_emb1, data.edge_index)
#     graph_node_sets.append(set(WL_emb2.tolist()))
#
# graph_emb_similarity = cal_graph_emb_similarity(graph_emb_list, 'cosine')
# graph_WL_similarity = cal_graph_WL_similarity(graph_node_sets)


# graph_num = mat1.shape[0]
# probe_score = 0
# for row in range(graph_num):
#     vec1 = mat1[row]
#     vec2 = mat2[row]
#     dij = torch.pow(vec1 - vec2, exponent=2)
#     dij_sum = torch.sum(dij)
#     probe_score += 1 - 6. * dij_sum / (graph_num * (graph_num ** 2 - 1))
#
# probe_score = probe_score / graph_num