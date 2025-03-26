import torch

from torch_geometric.nn import GCNConv


class GCN_test(torch.nn.Module):  # https://github.com/tkipf/pygcn
    def __init__(self, in_channels, hidden_channels, out_channels, hid_layers=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)

        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, node_emb, edge_index):
        node_emb = self.conv1(node_emb, edge_index)
        node_emb = self.relu(node_emb)
        node_emb = self.dropout(node_emb)
        return node_emb
