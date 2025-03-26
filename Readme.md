# Knowledge Probing for Graph Representation Learning

This repository contains the official implementation of the paper:

> **Knowledge Probing for Graph Representation Learning**  
> Mingyu Zhao, Xingyu Huang, Ziyu Lyu, Yanlin Wang, Lixin Cui, Lu Bai  
> *arXiv preprint arXiv:2408.03877*, 2024.

We propose a novel framework for probing knowledge in Graph Representation Learning (GRL). By leveraging various graph-based models, we design probing techniques to evaluate the extent to which learned embeddings capture structural and semantic information in graph data. Our approach is model-agnostic and supports multiple datasets and downstream tasks, enabling a systematic analysis of representational capacity.

---

## :package: Requirements

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

> :warning: Please ensure that you install the exact versions listed above to avoid compatibility issues.

---

## :rocket: Getting Started

All configurations are defined in `Utility/constant.py`. You can specify the model, dataset, and downstream task by modifying this file.

---

### :wrench: Key Configuration Parameters

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

### :file_folder: `List_permit_args` – Task Configuration List

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

### :mag: `Dict_probe_emb` – Probing Task Selection

This dictionary defines which downstream tasks (from `List_permit_args`) are used for each probing method.

Each probe (e.g., `CentProbe`, `DistProbe`) maps to a list of indices that correspond to entries in `List_permit_args`. The selected tasks will be evaluated using the specified probe.

For example:
```python
Dict_probe_emb = {
    'CentProbe':    np.arange(0, 2),  # Apply CentProbe on List_permit_args[0] and List_permit_args[1]
    'DistProbe':    np.arange(0, 2),
    'GstructProbe': np.arange(4, 6),
}
```

In this example:
- `CentProbe` and `DistProbe` will be applied to the first two tasks in `List_permit_args` (e.g., flickr and ppi).
- `GstructProbe` will be applied to tasks indexed 4 and 5 (e.g., link prediction or graph classification, depending on what’s uncommented).

To enable additional probes, you can modify or uncomment lines as needed.


---

### :arrow_forward: Running Experiments

1. **Uncomment** the desired configuration in `List_permit_args`.
2. **Specify** the model(s) in `List_GE_Model`.

#### Example

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

## :bookmark: Citation

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

Happy Probing! :dart:
