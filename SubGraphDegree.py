import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

dataname = "MUTAG"
data = torch.load('./Data/'+dataname+'/processed/data.pt')

def calculate_average_degree(data):
    edge_index = data[1]['edge_index']
    num_nodes = data[0].num_nodes

    node_degree = degree(data[0].edge_index[0], num_nodes)

    average_degrees = []
    for i in range(num_nodes):
        start_node = edge_index[0, i].item() 
        end_node = edge_index[1, i].item() 

        subgraph_degree = node_degree[start_node:end_node]
        average_degree = torch.mean(subgraph_degree.float())
        average_degrees.append(average_degree.item())

    return average_degrees

average_degrees = calculate_average_degree(data)
print(average_degrees)
