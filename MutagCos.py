import torch
from torch import cosine_similarity
from torch_geometric.utils import degree
import matplotlib.pyplot as plt

dataname = "MUTAG"
data = torch.load('./Data/'+dataname+'/processed/data.pt')

x = data[0].x
y = data[0].y

cos_sim_matrix = torch.zeros((y.size(0), y.size(0)))
cos_sim_matrix_diff = torch.zeros((y.size(0), y.size(0)))
total = 0
totalDifferent = 0
for i in range(y.size(0)):
    for j in range(i+1, y.size(0)):
        if y[i] == y[j]: 
            total += 1
            cos_sim = cosine_similarity(x[i].view(1, -1), x[j].view(1, -1))
            cos_sim_matrix[i][j] = cos_sim
            cos_sim_matrix[j][i] = cos_sim 
        else:
            totalDifferent += 1
            cos_sim_diff = cosine_similarity(x[i].view(1, -1), x[j].view(1, -1))
            cos_sim_matrix_diff[i][j] = cos_sim_diff
            cos_sim_matrix_diff[j][i] = cos_sim_diff

print(sum(sum(cos_sim_matrix))/total/2)
print(sum(sum(cos_sim_matrix_diff))/totalDifferent/2)