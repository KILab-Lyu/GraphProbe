from src.Utility.model.GCL.Utility.augmentors.augmentor import Graph, Augmentor
from src.Utility.model.GCL.Utility.augmentors.functional import permute


class NodeShuffling(Augmentor):
    def __init__(self):
        super(NodeShuffling, self).__init__()

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = permute(x)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
