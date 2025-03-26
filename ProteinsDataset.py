import pandas as pd
import networkx as nx
from Bio import SeqIO, SwissProt
from sklearn.feature_extraction.text import TfidfVectorizer

# 1.read uniprot_annotations
# uniprot_annotations = {}
# with open("/data2/myzhao/project/CodeRec/GNNProbe/new_datasets/UniProt/uniprot_sprot.dat", "r") as handle:
#     for record in SwissProt.parse(handle):
#         uniprot_id = record.entry_name
#         description = record.description
#         uniprot_annotations[uniprot_id] = description

# 2.read ppi_df
# ppi_df = pd.read_csv("9606.protein.links.v12.0.txt", sep=' ', usecols=['protein1',  'protein2', 'score'])

# 3.create graph
G = nx.Graph()

# read fasta file
for record in SeqIO.parse("/data2/myzhao/project/CodeRec/GNNProbe/new_datasets/UniProt/uniprot_sprot.fasta", "fasta"):
    uniprot_id = record.id.split('|')[1] 
    description = record.description
    sequence = str(record.seq)
    # process or store information


# add nodes and attributes
for uniprot_id, desc in uniprot_annotations.items():
    G.add_node(uniprot_id, description=desc)

# add edges
# for _, row in ppi_df.iterrows():
#     protein1 = row['protein1']
#     protein2 = row['protein2']
#     score = row['score']
#     if protein1 in uniprot_annotations and protein2 in uniprot_annotations:
#         G.add_edge(protein1, protein2, weight=score)

# 4.extract text description and convert to feature vector
descriptions = [data['description'] for node, data in G.nodes(data=True)]
vectorizer = TfidfVectorizer(max_features=1000)
description_features = vectorizer.fit_transform(descriptions).toarray()

# 5.add features to nodes
for i, (node, data) in enumerate(G.nodes(data=True)):
    G.nodes[node]['feature'] = description_features[i]

# 6.now, G contains PPI network and text description features of each protein, can be used for GNN training
