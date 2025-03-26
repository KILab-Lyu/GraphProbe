import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, MLP


class NodeClassifyTask(torch.nn.Module):
    def __init__(self, node_emb_dim, node_class_num, multi_label=False):
        super().__init__()
        self.linear = nn.Linear(in_features=node_emb_dim, out_features=node_class_num)

    def forward(self, emb, node_index=None):

        return self.linear(emb)


class LinkPredTask(torch.nn.Module):
    def __init__(self, node_emb_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=node_emb_dim, out_features=node_emb_dim)

    def forward(self, emb, edge_index):
        emb = self.linear(emb)
        out_node_index = edge_index[0].long()
        in_node_index = edge_index[1].long()
        pred = torch.zeros_like(out_node_index, dtype=torch.float16)
        i = 0
        pairs_per_loop = 30000000
        while i < len(pred):
            right_border = min(i + pairs_per_loop, len(pred))
            temp = emb[out_node_index[i:right_border]] * emb[in_node_index[i:right_border]]
            pred[i:right_border] = torch.mean(temp, dim=1)
            i += pairs_per_loop

        return pred.float().sigmoid()


class GraphClassifyTask(torch.nn.Module):
    def __init__(self, node_emb_dim, graph_class_num, readout='sum', hidden_layers=1):
        super().__init__()
        self.layers = MLP([node_emb_dim, node_emb_dim, graph_class_num], norm="batch_norm", act=None, dropout=0.5)
        # self.layers = nn.ModuleList([nn.Linear(node_emb_dim, node_emb_dim) for _ in range(hidden_layers)])

        # self.linear_out = nn.Linear(in_features=node_emb_dim, out_features=graph_class_num)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.elu = nn.ELU()

        if readout == 'max':
            self.readout = global_max_pool
        elif readout == 'sum':
            self.readout = global_add_pool
        elif readout == 'mean':
            self.readout = global_mean_pool
        else:
            print('invalid readout input kword')

    def forward(self, node_emb, batch):
        graph_emb = self.readout(node_emb, batch)
        graph_emb = self.layers(graph_emb)
        # graph_emb = self.dropout(graph_emb)

        # for layer in self.layers:
        #     graph_emb = layer(graph_emb)
        #     graph_emb = self.dropout(graph_emb)
        #     graph_emb = self.elu(graph_emb)

        # graph_emb = self.linear_out(graph_emb)
        return graph_emb


# class GraphClassifyTask1(torch.nn.Module):
#     def __init__(self, node_emb_dim, graph_class_num, readout='sum', hidden_layers=1):
#         super().__init__()
#         self.linear_out = nn.Linear(in_features=node_emb_dim, out_features=graph_class_num)
#
#         self.layers = nn.ModuleList([nn.Linear(node_emb_dim, node_emb_dim) for _ in range(hidden_layers)])
#
#         if readout == 'max':
#             self.readout = torch.max
#         elif readout == 'sum':
#             self.readout = torch.sum
#         elif readout == 'mean':
#             self.readout = torch.mean
#         else:
#             print('invalid readout input kword')
#
#     def forward(self, node_emb, batch):
#         if self.readout == torch.max:
#             graph_emb = self.readout(node_emb, dim=0).values
#         else:
#             graph_emb = self.readout(node_emb, dim=0)
#
#         for layer in self.layers:
#             graph_emb = layer(graph_emb)
#
#         pred = self.linear_out(graph_emb)
#         pred = pred.unsqueeze(dim=0)
#         return pred