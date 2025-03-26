# Knowledge Probing for Graph Representation Learning

This repository contains the implementation for the paper:

> **Knowledge Probing for Graph Representation Learning**  
> Mingyu Zhao, Xingyu Huang, Ziyu Lyu, Yanlin Wang, Lixin Cui, Lu Bai  
> *arXiv preprint arXiv:2408.03877*, 2024.

In this paper, we propose a novel method for probing knowledge in Graph Representation Learning (GRL). By leveraging graph-based models, we explore different probing techniques to assess the depth and breadth of knowledge captured in graph representations, especially focusing on structural and semantic relationships within graph data. Our approach can be applied to various datasets and downstream tasks to evaluate how well the learned graph embeddings encode meaningful relationships.

## Requirements

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

Please make sure to install the above versions to avoid compatibility issues.

## Usage

To run experiments with this code, adjust configurations in the `Utility/constant` file to specify the model, dataset, and downstream task.

For example, if you want to run the `Chebyshev` model on the `movielens` dataset with the `linkpredtask` downstream task, modify the respective entries in `constant` to reflect these choices.

Once configured, execute the following command:



```bash
python -u main.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhao2024knowledge,
  title={Knowledge Probing for Graph Representation Learning},
  author={Zhao, Mingyu and Huang, Xingyu and Lyu, Ziyu and Wang, Yanlin and Cui, Lixin and Bai, Lu},
  journal={arXiv preprint arXiv:2408.03877},
  year={2024}
}

9