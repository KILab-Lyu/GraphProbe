import torch
import torch.nn as nn
import json
import os
from collections import Counter

import torch_geometric
from dgl.data import YelpDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset, HeteroData
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.datasets import Planetoid, Yelp, MovieLens, Flickr, WikiCS, PPI, TUDataset
from torch_geometric.utils import subgraph, mask_to_index, index_to_mask, negative_sampling

# from src.Utility.constant import *


class PPIFullDataset(InMemoryDataset):
    def __init__(self, root='../Data/PPI'):
        self.data_list = []

        for split in ['train', 'val', 'test']:
            dataset = PPI(root=root, split=split)
            data_list = []

            for i, data in enumerate(dataset):
                data_list.append(data)

            self.data_list.extend(data_list)

        super(PPIFullDataset, self).__init__(root, None, None)
        self.data, self.slices = self.collate(self.data_list)


# class YelpDataset(Dataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(YelpDataset, self).__init__(root, transform, pre_transform)
#         dataset = Yelp(root=root)
#         self.data = dataset.data
#
#         # self.edge_index = self.data.edge_index
#         self.num_nodes = self.data.num_nodes
#
#         self.node_perm = torch.arange(self.data.num_nodes)
#         self.batch_size = 16384
#
#     def len(self):
#         return self.num_nodes // self.batch_size + 1
#
#     def get(self, item):
#         data = self.data.clone()
#         start = item * self.batch_size
#         end = min((item+1) * self.batch_size, self.num_nodes)
#
#         sg_node_index = torch.arange(start, end)
#         sg_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
#         sg_x = data.x[sg_node_index]
#         sg_y = data.y[sg_node_index]
#
#         return Data(x=sg_x, edge_index=sg_edge_index, y=sg_y)


# class YelpOriginDataset(Dataset):
#     def __init__(self, root='../Data/Yelp', transform=None, pre_transform=None):
#         super(YelpOriginDataset, self).__init__(root, transform, pre_transform)
#         user_id_set = set()
#         with open(os.path.join(root, "yelp_academic_dataset_user.json"), encoding='utf8') as json_file:
#             for line in json_file:
#                 line_contents = json.loads(line)
#                 if line_contents['review_count'] > 40:
#                     user_id_set.add(line_contents['user_id'])
#
#         self.num_user = len(user_id_set)
#         user_id_dict = dict(zip(user_id_set, np.arange(self.num_user)))
#
#         feature = []
#         # item_categories = set()
#         item_all_feature = []
#         item_id_set = set()
#         with open(os.path.join(root, "yelp_academic_dataset_business.json"), encoding='utf8') as json_file:
#             for line in json_file:
#                 line_contents = json.loads(line)
#                 if line_contents['review_count'] > 40 and line_contents['categories'] is not None:
#                     item_id_set.add(line_contents['business_id'])
#                     feat = line_contents['categories'].split(',')
#                     # item_categories.update(feat)
#                     feature.append(feat)
#                     item_all_feature.extend(feat)
#
#         self.num_item = len(item_id_set)
#         self.num_nodes = self.num_item + self.num_user
#         item_id_dict = dict(zip(item_id_set, np.arange(self.num_item)+self.num_user))
#
#         item_feature = []
#         counter = Counter(item_all_feature)
#         item_categories = counter.most_common(64)
#         feat_dim = len(item_categories)
#         item_categories = {c[0]: i for i, c in enumerate(item_categories)}
#         for i, c in enumerate(feature):
#             empty = np.zeros(feat_dim)
#             loc = set(map(lambda x: item_categories.get(x, None), c))
#             loc.discard(None)
#             empty[list(loc)] = 1
#             item_feature.append(empty)
#         item_feature = torch.from_numpy(np.stack(item_feature, axis=1)).t()
#
#         emb = nn.Embedding(num_embeddings=self.num_user, embedding_dim=feat_dim)
#         emb = emb(torch.arange(self.num_user, dtype=torch.long))
#         x = torch.cat([emb, item_feature], dim=0)
#         x = x.float()
#
#         edge = []
#         missed_node = []
#         with open(os.path.join(root, "yelp_academic_dataset_tip.json"), encoding='utf8') \
#                 as json_file:
#             for line in json_file:
#                 line_contents = json.loads(line)
#                 user = user_id_dict.get(line_contents['user_id'], None)
#                 item = item_id_dict.get(line_contents['business_id'], None)
#                 if user is None or item is None:
#                     missed_node.append((user, item))
#                 else:
#                     edge.append([user, item])
#
#         edge_index = torch.tensor(edge, dtype=torch.long).t()
#
#         self.data = Data(edge_index=edge_index, x=x)
#
#         if os.path.exists(os.path.join(root, "randperm.pt")):
#             randperm = torch.load(os.path.join(root, "randperm.pt"))
#         else:
#             randperm = torch.randperm(self.num_item)
#             torch.save(randperm, os.path.join(root, "randperm.pt"))
#         item_list = torch.arange(self.num_user, self.num_nodes)[randperm]
#         self.train_item_list = item_list[:int(self.num_item * 0.5)]
#         self.test_item_list = item_list[int(self.num_item * 0.5):]
#
#         self.batch_size = 2048
#         self.is_train = True
#
#     def len(self):
#         return self.num_user // self.batch_size + 1
#
#     def get(self, item):
#         data = self.data.clone()
#         start = item * self.batch_size
#         end = min((item+1) * self.batch_size, self.num_user)
#
#         item_list = self.train_item_list if self.is_train else self.test_item_list
#         sg_node_index = torch.cat([torch.arange(start, end), item_list])
#         sg_pos_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
#         sg_neg_edge_index = negative_sampling(sg_pos_edge_index, sg_node_index.shape[0])
#
#         sg_x = data.x[sg_node_index]
#
#         sg_data = Data(x=sg_x, edge_index=sg_pos_edge_index)
#         sg_data['neg_edge_index'] = sg_neg_edge_index
#         return sg_data


class FlickrDataset(Dataset):
    def __init__(self, root='./Data/Flickr/', transform=None, pre_transform=None):
        super(FlickrDataset, self).__init__(root, transform, pre_transform)
        dataset = Flickr(root=root)
        self.data = dataset.data

        self.num_nodes = self.data.num_nodes

        if os.path.exists(os.path.join(root, "randperm.pt")):
            randperm = torch.load(os.path.join(root, "randperm.pt"))
        else:
            randperm = torch.arange(self.data.num_nodes)
            torch.save(randperm, os.path.join(root, "randperm.pt"))
        self.node_perm = randperm
        self.batch_size = 4096

    def len(self):
        return self.num_nodes // self.batch_size + 1

    def get(self, item):
        data = self.data.clone()
        start = item * self.batch_size
        end = min((item+1) * self.batch_size, self.num_nodes)

        sg_node_index = torch.arange(start, end)
        sg_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
        sg_x = data.x[sg_node_index]
        sg_y = data.y[sg_node_index]

        return Data(x=sg_x, edge_index=sg_edge_index, y=sg_y)

from torch_geometric.utils import negative_sampling

class FlickrDataset1(Dataset):
    def __init__(self, root='./Data/Flickr/', transform=None, pre_transform=None):
        super(FlickrDataset1, self).__init__(root, transform, pre_transform)
        dataset = Flickr(root=root)
        self.data = dataset.data

        self.num_nodes = self.data.num_nodes

        if os.path.exists(os.path.join(root, "randperm.pt")):
            randperm = torch.load(os.path.join(root, "randperm.pt"))
        else:
            randperm = torch.arange(self.data.num_nodes)
            torch.save(randperm, os.path.join(root, "randperm.pt"))
        self.node_perm = randperm
        self.batch_size = 4096

    def len(self):
        return self.num_nodes // self.batch_size + 1

    def get(self, item):
        data = self.data.clone()
        start = item * self.batch_size
        end = min((item+1) * self.batch_size, self.num_nodes)

        sg_node_index = torch.arange(start, end)
        sg_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
        sg_x = data.x[sg_node_index]
        sg_y = data.y[sg_node_index]


        sg_neg_edge_index = negative_sampling(sg_edge_index, sg_node_index.shape[0])

        sg_data = Data(x=sg_x, edge_index=sg_edge_index, y=sg_y)
        sg_data['neg_edge_index'] = sg_neg_edge_index
        return sg_data

class Yelp2018Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Yelp2018Dataset, self).__init__(root, transform, pre_transform)
        dataset = torch_geometric.datasets.Yelp(root='./Data/Yelp/')
        self.num_item = dataset.data.x.shape[0]
        self.num_user = dataset.data.num_nodes
        self.num_nodes = self.num_item + self.num_user
        edge_index = dataset.data.edge_index
        edge_index[1] = edge_index[1] + self.num_user


        emb = nn.Embedding(num_embeddings=self.num_user, embedding_dim=404)

        emb = emb(torch.arange(self.num_user, dtype=torch.long))

        x = torch.empty((self.num_nodes, 200))

        self.data = Data(edge_index=edge_index, x=x)

        if os.path.exists(os.path.join(root, "randperm.pt")):
            randperm = torch.load(os.path.join(root, "randperm.pt"))
        else:
            randperm = torch.randperm(self.num_item)
            torch.save(randperm, os.path.join(root, "randperm.pt"))

        item_list = torch.arange(self.num_user, self.num_nodes)[randperm]
        self.train_item_list = item_list[:int(self.num_item * 0.7)]
        self.test_item_list = item_list[int(self.num_item * 0.7):]

        self.batch_size = 512
        self.is_train = True

    def len(self):
        return self.num_user // self.batch_size + 1

    def get(self, item):
        data = self.data.clone()
        start = item * self.batch_size
        end = min((item+1) * self.batch_size, self.num_user)
        item_list = self.train_item_list if self.is_train else self.test_item_list
        sg_node_index = torch.cat([torch.arange(start, end), item_list])
        sg_pos_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
        sg_neg_edge_index = negative_sampling(sg_pos_edge_index, sg_node_index.shape[0])

        sg_x = data.x[sg_node_index]

        sg_data = Data(x=sg_x, edge_index=sg_pos_edge_index)
        sg_data['neg_edge_index'] = sg_neg_edge_index
        return sg_data

class MovielensDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MovielensDataset, self).__init__(root, transform, pre_transform)
        dataset = MovieLens(root=root)
        self.num_item = dataset.data['movie'].x.shape[0]
        self.num_user = dataset.data['user'].num_nodes
        self.num_nodes = self.num_item + self.num_user
        edge_index = dataset.data[('user', 'rates', 'movie')]['edge_index']
        edge_index[1] = edge_index[1] + self.num_user

        emb = nn.Embedding(num_embeddings=self.num_user, embedding_dim=404)
        emb = emb(torch.arange(self.num_user, dtype=torch.long))
        x = torch.empty((self.num_nodes, 404))
        x[self.num_user:] = dataset.data['movie'].x
        x[:self.num_user] = emb

        self.data = Data(edge_index=edge_index, x=x)

        if os.path.exists(os.path.join(root, "randperm.pt")):
            randperm = torch.load(os.path.join(root, "randperm.pt"))
        else:
            randperm = torch.randperm(self.num_item)
            torch.save(randperm, os.path.join(root, "randperm.pt"))
        item_list = torch.arange(self.num_user, self.num_nodes)[randperm]
        self.train_item_list = item_list[:int(self.num_item * 0.7)]
        self.test_item_list = item_list[int(self.num_item * 0.7):]

        self.batch_size = 512
        self.is_train = True

    def len(self):
        return self.num_user // self.batch_size + 1

    def get(self, item):
        data = self.data.clone()
        start = item * self.batch_size
        end = min((item+1) * self.batch_size, self.num_user)

        item_list = self.train_item_list if self.is_train else self.test_item_list
        sg_node_index = torch.cat([torch.arange(start, end), item_list])
        sg_pos_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
        sg_neg_edge_index = negative_sampling(sg_pos_edge_index, sg_node_index.shape[0])

        sg_x = data.x[sg_node_index]

        sg_data = Data(x=sg_x, edge_index=sg_pos_edge_index)
        sg_data['neg_edge_index'] = sg_neg_edge_index
        return sg_data

class AmazonBookDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AmazonBookDataset, self).__init__(root, transform, pre_transform)
        self.num_user = 52643
        self.num_item = 91599
        self.num_nodes = self.num_user + self.num_item

        with open('./Data/amazon-book/train.txt', "r") as f:
            train_line = f.readlines()
            train_edges = []
            for i in range(len(train_line)):
                nodes = train_line[i].split()
                for j in range(1, len(nodes)):
                    train_edges.append((int(nodes[0]), int(nodes[j]) + self.num_user))

        with open('./Data/amazon-book/test.txt', "r") as f:
            test_line = f.readlines()
            test_edges = []
            for i in range(len(test_line)):
                nodes = test_line[i].split()
                for j in range(1, len(nodes)):
                    test_edges.append((int(nodes[0]), int(nodes[j]) + self.num_user))

        edge_index = torch.tensor(train_edges + test_edges).t()
        self.data = Data(edge_index=edge_index)
        self.data.num_nodes = self.num_nodes

        import numpy as np
        self.train_item_list = torch.tensor(list(set(np.array(train_edges)[:, 1])))
        self.test_item_list = torch.tensor(list(set(np.array(test_edges)[:, 1])))

        self.batch_size = 8192

        self.is_train = True

    def len(self):
        return self.num_user // self.batch_size + 1

    def get(self, item):
        data = self.data.clone()
        start = item * self.batch_size
        end = min((item+1) * self.batch_size, self.num_user)

        item_list = self.train_item_list if self.is_train else self.test_item_list
        sg_node_index = torch.cat([torch.arange(start, end), item_list])
        sg_pos_edge_index, _ = subgraph(sg_node_index, data.edge_index, relabel_nodes=True)
        sg_neg_edge_index = negative_sampling(sg_pos_edge_index, sg_node_index.shape[0])

        sg_x = data.x[sg_node_index]

        sg_data = Data(x=sg_x, edge_index=sg_pos_edge_index)
        sg_data['neg_edge_index'] = sg_neg_edge_index
        return sg_data

# cora = Planetoid(root='./Data/', name='Cora')
# citeSeer = Planetoid(root='./Data/', name='Citeseer')
# pubMed = Planetoid(root='./Data/', name='PubMed')

# yelp = YelpDataset(root='../Data/Yelp/')
# yelp2018 = Yelp2018Dataset(root='./Data/Yelp2018/')
# flickr = FlickrDataset1(root='./Data/Flickr/')
# AmazonBook = AmazonBookDataset(root="./Data/amazon-book/")

# yelp = YelpOriginDataset()
# movielens = MovielensDataset(root='./Data/MovieLens/')
#
# ppi = PPIFullDataset(root='../Data/PPI')
# mutag = TUDataset(root='../Data/', name='MUTAG', use_node_attr=True)
# proteins = TUDataset(root='../Data/', name='PROTEINS', use_node_attr=True)
# enzymes = TUDataset(root='../Data/', name='ENZYMES', use_node_attr=True)
#
# wikics = WikiCS(root='../Data/WikiCS/', is_undirected=False)
# imdb = TUDataset(root='../Data/', name='IMDB-BINARY', use_node_attr=True)
# flickr = Flickr(root='../Data/Flickr/')

# import torch_geometric.transforms as T
# dataset = Planetoid(root='/tmp/' + data_name, name=data_name)
# if normalize_features:
#     dataset.transform = T.NormalizeFeatures()


def select_data(data_name, node_feat_dim=None):

    dataset = {}
    if data_name in ('Cora', 'Citeseer', 'PubMed'  ):
        dataset = Planetoid(root='./Data/', name=data_name)
    elif data_name == 'cora':
        cora = Planetoid(root='./Data/', name='Cora')
        dataset = cora
    elif data_name == 'citeseer':
        citeSeer = Planetoid(root='./Data/', name='Citeseer')
        dataset = citeSeer
    elif data_name == 'pubmed':
        pubMed = Planetoid(root='./Data/', name='PubMed')
        dataset = pubMed

    elif data_name == 'yelp':
        yelp2018 = Yelp2018Dataset(root='./Data/Yelp2018/')
        dataset = yelp2018
    elif data_name == 'flickr':
        flickr = FlickrDataset1(root='./Data/Flickr/')
        dataset = flickr
    elif data_name == 'movielens':
        movielens = MovielensDataset(root='./Data/MovieLens/')
        dataset = movielens
    elif data_name == 'amazon':
        AmazonBook = AmazonBookDataset(root="./Data/amazon-book/")
        dataset = AmazonBook

    # elif data_name == 'ppi':
    #     dataset = ppi
    elif data_name == 'mutag':
        mutag = TUDataset(root='./Data/', name='MUTAG', use_node_attr=True)
        dataset = mutag
    # elif data_name == 'proteins':
    #     dataset = proteins
    elif data_name == 'enzymes':
        enzymes = TUDataset(root='./Data/', name='ENZYMES', use_node_attr=True)
        dataset = enzymes
    dataset.name = data_name

    if node_feat_dim == False:
        emb = nn.Embedding(num_embeddings=dataset.data.num_nodes, embedding_dim=node_feat_dim)
        emb = emb(torch.arange(dataset.data.num_nodes, dtype=torch.long, requires_grad=False))
        # emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
        flag = dataset.data.update_tensor(emb.detach(), x_attr)
    else:
        if data_name == 'yelp':
            x = dataset.data.x
            x = torch.nn.functional.normalize(x,p=2,dim=1)
            dataset.data.x = x.detach()
        else:
            x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
            x = dataset.data.get_tensor(x_attr)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            x_attr = TensorAttr(group_name=None, attr_name='x', index=None)
            dataset.data.update_tensor(x.detach(), x_attr)
    return dataset

def apply_node_perturbation(dataset, perturbation_ratio=0.01, data_name=None):
    edge_index = dataset.data.edge_index
    num_nodes = dataset.data.x.shape[0]
    num_edges = edge_index.shape[1]
    num_edge_changes = int(num_edges * perturbation_ratio)

    edges_to_remove = random.sample(range(num_edges), num_edge_changes)
    edge_index = edge_index[:, [i for i in range(num_edges) if i not in edges_to_remove]]

    for _ in range(num_edge_changes):
        node_a, node_b = random.sample(range(num_nodes), 2)
        edge_index = torch.cat((edge_index, torch.tensor([[node_a], [node_b]])), dim=1)

    nodes_to_remove = random.sample(range(num_nodes), int(num_nodes * perturbation_ratio))
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[nodes_to_remove] = False
    dataset.data.x = dataset.data.x[mask]

    if data_name != "movielens":
        dataset.data.y = dataset.data.y[mask]
        dataset.data.train_mask = dataset.data.train_mask[mask]
        dataset.data.val_mask = dataset.data.val_mask[mask]
        dataset.data.test_mask = dataset.data.test_mask[mask]

    valid_edges = []
    for i in range(edge_index.shape[1]):
        if mask[edge_index[0, i]] and mask[edge_index[1, i]]:
            valid_edges.append(i)

    dataset.data.edge_index = edge_index[:, valid_edges]

    new_num_nodes = dataset.data.x.shape[0]

    for i in range(dataset.data.edge_index.shape[1]):
        dataset.data.edge_index[0, i] = min(dataset.data.edge_index[0, i], new_num_nodes - 1)
        dataset.data.edge_index[1, i] = min(dataset.data.edge_index[1, i], new_num_nodes - 1)

    dataset.data.edge_index = dataset.data.edge_index[:, dataset.data.edge_index[0] < new_num_nodes]

    return dataset

import os
import torch
import random

def apply_node_perturbation_movielens(dataset, perturbation_ratio=0.01):

    edge_index = dataset.data.edge_index
    num_edges = edge_index.shape[1]
    num_edge_changes = int(num_edges * perturbation_ratio)

    edges_to_remove = set(random.sample(range(num_edges), num_edge_changes))
    keep_edge_indices = [i for i in range(num_edges) if i not in edges_to_remove]
    new_edge_index = edge_index[:, keep_edge_indices]

    for _ in range(num_edge_changes):
        user = random.randint(0, dataset.num_user - 1)
        movie = random.randint(dataset.num_user, dataset.num_nodes - 1)
        new_edge = torch.tensor([[user], [movie]], dtype=new_edge_index.dtype)
        new_edge_index = torch.cat([new_edge_index, new_edge], dim=1)

    dataset.data.edge_index = new_edge_index

    num_movie_nodes = dataset.num_item
    num_movie_removals = int(num_movie_nodes * perturbation_ratio)

    if num_movie_removals < 1 and perturbation_ratio > 0:
        num_movie_removals = 1

    movie_indices = list(range(dataset.num_user, dataset.num_nodes))
    nodes_to_remove = set(random.sample(movie_indices, num_movie_removals))

    full_mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
    for idx in nodes_to_remove:
        full_mask[idx] = False

    dataset.data.x = dataset.data.x[full_mask]

    old_movie_indices = torch.arange(dataset.num_user, dataset.num_nodes)
    movie_mask = full_mask[dataset.num_user:]
    remaining_old_movie_indices = old_movie_indices[movie_mask]
    new_movie_indices = torch.arange(dataset.num_user, dataset.num_user + remaining_old_movie_indices.shape[0])
    mapping_movie = {old.item(): new.item() for old, new in zip(remaining_old_movie_indices, new_movie_indices)}

    old_train = dataset.train_item_list.tolist()
    new_train = [mapping_movie[x] for x in old_train if x in mapping_movie]
    dataset.train_item_list = torch.tensor(new_train, dtype=torch.long)

    old_test = dataset.test_item_list.tolist()
    new_test = [mapping_movie[x] for x in old_test if x in mapping_movie]
    dataset.test_item_list = torch.tensor(new_test, dtype=torch.long)

    mapping_all = {i: i for i in range(dataset.num_user)} 
    mapping_all.update(mapping_movie) 

    old_edge_index = dataset.data.edge_index
    new_edges = []
    for i in range(old_edge_index.shape[1]):
        u = old_edge_index[0, i].item()
        v = old_edge_index[1, i].item()
        if u in mapping_all and v in mapping_all:
            new_u = mapping_all[u]
            new_v = mapping_all[v]
            new_edges.append([new_u, new_v])
    if len(new_edges) > 0:
        new_edge_index = torch.tensor(new_edges, dtype=old_edge_index.dtype).t().contiguous()
    else:
        new_edge_index = torch.empty((2, 0), dtype=old_edge_index.dtype)
    dataset.data.edge_index = new_edge_index

    dataset.num_item = len(mapping_movie)
    dataset.num_nodes = dataset.num_user + dataset.num_item

    return dataset

from torch_geometric.data import Data

# def apply_node_perturbation_proteins_mutag(dataset, perturbation_ratio=0.01):
#     edge_index = dataset.data.edge_index
#     num_edges = edge_index.shape[1]
#     num_edge_changes = int(num_edges * perturbation_ratio)
#
#     edges_to_remove = set(random.sample(range(num_edges), num_edge_changes))
#     keep_edge_indices = [i for i in range(num_edges) if i not in edges_to_remove]
#     new_edge_index = edge_index[:, keep_edge_indices]
#
#     num_nodes = dataset.data.x.shape[0]
#     for _ in range(num_edge_changes):
#         node_a, node_b = random.sample(range(num_nodes), 2)
#         new_edge = torch.tensor([[node_a], [node_b]], dtype=new_edge_index.dtype)
#         new_edge_index = torch.cat([new_edge_index, new_edge], dim=1)
#
#     dataset.data.edge_index = new_edge_index
#
#     num_nodes = dataset.data.x.shape[0]
#     num_nodes_to_remove = int(num_nodes * perturbation_ratio)
#     if num_nodes_to_remove < 1 and perturbation_ratio > 0:
#         num_nodes_to_remove = 1
#
#     nodes_to_remove = set(random.sample(range(num_nodes), num_nodes_to_remove))
#
#     full_mask = torch.ones(num_nodes, dtype=torch.bool)
#     for idx in nodes_to_remove:
#         full_mask[idx] = False
#
#     dataset.data.x = dataset.data.x[full_mask]

#     if hasattr(dataset.data, 'y'):
#         dataset.data.y = dataset.data.y[full_mask]
#
#     if hasattr(dataset.data, 'train_mask'):
#         dataset.data.train_mask = dataset.data.train_mask[full_mask]
#     if hasattr(dataset.data, 'val_mask'):
#         dataset.data.val_mask = dataset.data.val_mask[full_mask]
#     if hasattr(dataset.data, 'test_mask'):
#         dataset.data.test_mask = dataset.data.test_mask[full_mask]

#     valid_edges = []
#     for i in range(new_edge_index.shape[1]):
#         u = new_edge_index[0, i].item()
#         v = new_edge_index[1, i].item()
#         if full_mask[u] and full_mask[v]:
#             valid_edges.append(i)

#     dataset.data.edge_index = new_edge_index[:, valid_edges]
#
#     dataset.num_nodes = dataset.data.x.shape[0]
#
#     return dataset
