# Knowledge Probing for Graph Representation Learning

This repository contains the official implementation of the paper:

> **Knowledge Probing for Graph Representation Learning**  
> Mingyu Zhao, Xingyu Huang, Ziyu Lyu, Yanlin Wang, Lixin Cui, Lu Bai  
> *arXiv preprint arXiv:2408.03877*, 2024.

We propose a novel framework for probing knowledge in Graph Representation Learning (GRL). By leveraging various graph-based models, we design probing techniques to evaluate the extent to which learned embeddings capture structural and semantic information in graph data. Our approach is model-agnostic and supports multiple datasets and downstream tasks, enabling a systematic analysis of representational capacity.

---

## ?? Requirements

The code has been tested with the following package versions:

- `torch`: 2.1.0+cu118  
- `torch-cluster`: 1.6.2+pt21cu118  
- `torch-geometric`: 2.6.1  
- `torch-scatter`: 2.1.2+pt21cu118  
- `torch-sparse`: 0.6.18+pt21cu118  
- `torch-spline-conv`: 1.2.2+pt21cu118  
- `torchaudio`: 2.1.0+cu118  
- `torchdata`: 0.7.0  
- `torchvision`: 0.16.0+cu118  
- `dgl`: 2.3.0+cu118

> ?? Please ensure that you install the exact versions listed above to avoid compatibility issues.

---

## ?? Getting Started

All configurations are defined in `Utility/constant.py`. You can specify the model, dataset, and downstream task by modifying this file.

---

### ?? Key Configuration Parameters

#### `device`
Sets the computing device:
```python
device = torch.device('cuda')
```

#### `TRANSDUCTIVE` and `INDUCTIVE`
Define the learning modes:
```python
TRANSDUCTIVE = 0  # For transductive learning
INDUCTIVE = 1     # For inductive learning
```

#### `EMB_ROOT`
Path to store the learned graph embeddings:
```python
EMB_ROOT = './Embedding'
```

#### `List_GE_Model`
A list of graph embedding models to run. Example:
```python
List_GE_Model = [
    'GCN',
    'Node2Vec',
    ...
]
```

#### `meta_feat`
Meta-level feature embeddings passed from command-line arguments:
```python
meta_feat = args.meta_emb
```

---

### ?? `List_permit_args` – Task Configuration List

Defines all allowed task settings. Each entry is a dictionary that specifies:

- `data`: Dataset name (`cora`, `pubmed`, `ppi`, `flickr`, etc.)
- `downstreamTask`: Downstream task type (`NodeClassifyTask`, `LinkPredTask`, etc.)
- `mode`: Learning mode (`TRANSDUCTIVE` or `INDUCTIVE`)
- `node_class_num` / `graph_class_num`: Number of classes
- `multilabel`: Whether it's a multi-label classification task
- `meta_feat`: Meta feature embeddings

**Example:**
```python
{
  'data': 'ppi',
  'downstreamTask': 'NodeClassifyTask',
  'mode': INDUCTIVE,
  'node_class_num': 121,
  'multilabel': True,
  'meta_feat': meta_feat
}
```

---

### ?? `Dict_probe_emb` – Probe Feature Dimensions

Defines the embedding dimensions used by different probe types:
```python
Dict_probe_emb = {
    'CentProbe':    np.arange(0, 2),
    'DistProbe':    np.arange(0, 2),
    'GstructProbe': np.arange(4, 6),
}
```

---

### ?? Running Experiments

1. **Uncomment** the desired configuration in `List_permit_args`.
2. **Specify** the model(s) in `List_GE_Model`.

#### ? Example

To run inductive node classification on the `ppi` dataset using `GCN`:

1. In `List_permit_args`, uncomment:
```python
{
  'data': 'ppi',
  'downstreamTask': 'NodeClassifyTask',
  'mode': INDUCTIVE,
  'node_class_num': 121,
  'multilabel': True,
  'meta_feat': meta_feat
}
```

2. In `List_GE_Model`, set:
```python
List_GE_Model = [
    'GCN',
]
```

3. Then execute the main script:
```bash
python -u main.py
```

---

## ?? Citation

If you find this work helpful, please consider citing:

```bibtex
@article{zhao2024knowledge,
  title={Knowledge Probing for Graph Representation Learning},
  author={Zhao, Mingyu and Huang, Xingyu and Lyu, Ziyu and Wang, Yanlin and Cui, Lixin and Bai, Lu},
  journal={arXiv preprint arXiv:2408.03877},
  year={2024}
}
```

---

Happy Probing! ??
