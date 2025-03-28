import numpy as np
import torch
from Utility.parser import args

device = torch.device('cuda')

TRANSDUCTIVE = 0
INDUCTIVE = 1

EMB_ROOT = './Embedding'

List_GE_Model = [
    'Chebyshev',
    'GCN',
    'GAT',
    # 'GraphSAGE',
    # 'VGAE',
    # 'GCL',
    # 'WGCN',
    # 'Node2Vec',
    # 'MLP',
    # 'LightGCN',
    # 'SSGCN',
    # 'GIN',
    # 'DeepWalk',
    # 'GraphRNN',
    # 'LightGCL',
    # 'SGL',
    # 'GCNTest',
    # 'GCNTest_agg',
    # 'GNN',
    # 'VGAE-GNN',
    #　''
]
# 'Node2Vec' 'DeepWalk' 'TAG',


meta_feat = args.meta_emb #


List_permit_args = [
    # transductive-NC 0-3
    # {'data': 'cora', 'downstreamTask': 'NodeClassifyTask', 'mode': TRANSDUCTIVE, 'node_class_num': 7, 'multilabel': False, 'meta_feat': meta_feat},
    # {'data': 'citeseer', 'downstreamTask': 'NodeClassifyTask', 'mode': TRANSDUCTIVE, 'node_class_num': 6, 'multilabel': False, 'meta_feat': meta_feat},
    # {'data': 'pubmed', 'downstreamTask': 'NodeClassifyTask', 'mode': TRANSDUCTIVE, 'node_class_num': 3, 'multilabel': False, 'meta_feat': meta_feat},
    # {'data': 'cora', 'downstreamTask':'NodeClassifyTask', 'mode':TRANSDUCTIVE, 'node_class_num':7, 'multilabel':False, 'meta_feat':meta_feat},
    # {'data': 'movielens', 'downstreamTask':'NodeClassifyTask', 'mode':TRANSDUCTIVE, 'node_class_num':7, 'multilabel':False, 'meta_feat':meta_feat},
    #
    # {'data': 'yelp', 'downstreamTask':'NodeClassifyTask', 'mode':TRANSDUCTIVE, 'node_class_num':7, 'multilabel':False, 'meta_feat':meta_feat},


    # transductive-GR 3-6
    # {'data': 'cora', 'downstreamTask': 'GraphReconsTask', 'mode': TRANSDUCTIVE, 'meta_feat': meta_feat},
    {'data': 'citeseer', 'downstreamTask': 'GraphReconsTask', 'mode': TRANSDUCTIVE, 'meta_feat': meta_feat},
    # {'data': 'pubmed', 'downstreamTask': 'GraphReconsTask', 'mode': TRANSDUCTIVE, 'meta_feat': meta_feat},

    # inductive-NC 3-5
    {'data': 'flickr', 'downstreamTask': 'NodeClassifyTask', 'mode': INDUCTIVE, 'node_class_num': 7, 'multilabel': False, 'meta_feat': meta_feat},
    # {'data': 'ppi', 'downstreamTask': 'NodeClassifyTask', 'mode': INDUCTIVE, 'node_class_num': 121, 'multilabel': True, 'meta_feat': meta_feat},
    #
    # inductive-LP 5-7
    # {'data': 'movielens', 'downstreamTask': 'LinkPredTask', 'mode': INDUCTIVE, 'meta_feat': meta_feat},
    # {'data': 'flickr', 'downstreamTask': 'LinkPredTask', 'mode': INDUCTIVE, 'meta_feat': meta_feat},
    # {'data': 'yelp', 'downstreamTask': 'LinkPredTask', 'mode': INDUCTIVE, 'meta_feat': meta_feat},
    # {'data': 'amazon', 'downstreamTask': 'LinkPredTask', 'mode': INDUCTIVE, 'meta_feat': meta_feat},

    # inductive-GC 7-9
    {'data': 'mutag', 'downstreamTask': 'GraphClassifyTask', 'mode': INDUCTIVE, 'graph_class_num': 2, 'meta_feat': meta_feat},
    # {'data': 'proteins', 'downstreamTask': 'GraphClassifyTask', 'mode': INDUCTIVE, 'graph_class_num': 2, 'meta_feat': meta_feat},
    {'data': 'enzymes', 'downstreamTask': 'GraphClassifyTask', 'mode': INDUCTIVE, 'graph_class_num': 6, 'meta_feat': meta_feat},
]

Dict_probe_emb = {
    'CentProbe':        np.arange(0, 2),
    'DistProbe':        np.arange(0, 2),
    'GstructProbe':     np.arange(2, 4),
    # 'ClusterProbe':     np.concatenate([np.arange(0, 3), np.arange(7, 9)]),
    # 'ContrastiveProbe':    np.concatenate([np.arange(0, 3), np.arange(7, 9)]),
}