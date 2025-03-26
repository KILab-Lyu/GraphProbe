import torch
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
dataname = "MUTAG"

data = torch.load('./Data/'+dataname+'/processed/data.pt')
def calculate_homophily(data):
    edge_index = data[0].edge_index
    labels = data[0].y

    neighbors_labels = {i: [] for i in range(labels.size(0))}

    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        neighbors_labels[source].append(labels[target].item())
        neighbors_labels[target].append(labels[source].item())

    homophily_values = []
    for i in range(labels.size(0)):
        if len(neighbors_labels[i]) == 0:
            homophily_values.append(0)
        else:
            same_label_count = neighbors_labels[i].count(labels[i].item())
            homophily_value = same_label_count / len(neighbors_labels[i])
            homophily_values.append(homophily_value)

    return homophily_values

homophily_values = calculate_homophily(data)
print(sum(homophily_values)/len(homophily_values))

