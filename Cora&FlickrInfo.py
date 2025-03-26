import torch
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
dataname = "MUTAG"

data = torch.load('./Data/'+dataname+'/processed/data.pt')
if dataname is "MovieLens":
    edge_index = data[0]['user', 'rates', 'movie'].edge_index
else:
    edge_index = data[0].edge_index
degree_list = degree(edge_index[0]).tolist()

max_degree = max(degree_list)
min_degree = min(degree_list)
avg_degree = sum(degree_list) / len(degree_list)

print("Cur dataset is {}".format(dataname))

print(f'max degree: {max_degree}')
print(f'min degree: {min_degree}')
print(f'avg degree: {avg_degree:.2f}')

plt.hist(degree_list, bins=range(1, 40, 5), edgecolor='black')
plt.title('The Degree of {}'.format(dataname))
plt.xlabel('Degree')
plt.ylabel('Total')
plt.savefig("{}Info.pdf".format(dataname))
# plt.show()
from torch_geometric.utils import homophily
homophily_ratio = homophily(edge_index, data[0].y[0])
print(f'{dataname} homophily ratio: {homophily_ratio:.4f}')
