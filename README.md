### Graph Neural Networks for Node Classification on OGBN-Arxiv

**Group 07** · SLTC Research University
Lecturer in Charge: Dr. Chameera De Silva · Teaching Assistant: Mr. Chamod Hewage

---

## Overview

This project implements and compares two Graph Neural Network architectures — a **Graph Convolutional Network (GCN)** and a **Graph Attention Network (GAT)** — for node classification on the **OGBN-Arxiv** citation network from the Open Graph Benchmark.

Each node in the graph represents a research paper, and each directed edge represents one paper citing another. The task is to predict which of **40 subject categories** a paper belongs to, using both its own content (a 128-dimensional feature vector derived from its title and abstract) and its position within the citation graph.

The project covers the full pipeline: tensor fundamentals, graph structural analysis, data preparation, GNN model development, training and optimization, evaluation, explainability, and an interactive Streamlit dashboard.

## Dataset

| Property | Value |
|---|---|
| Nodes (papers) | 169,343 |
| Edges (citations) | 1,166,243 |
| Node feature dimension | 128 |
| Number of classes | 40 subject categories |
| Graph density | 0.000041 |
| Train / Val / Test split | 90,941 / 29,799 / 48,603 (official OGB split, by publication year) |

## Results

| Model | Split | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| GCN | Validation | 0.5835 | 0.3905 | 0.3072 | 0.3167 |
| **GCN** | **Test** | **0.5254** | **0.3514** | **0.2869** | **0.2836** |
| GAT | Validation | 0.5601 | 0.3702 | 0.2391 | 0.2440 |
| GAT | Test | 0.5125 | 0.3296 | 0.2297 | 0.2238 |

GCN's simpler, fixed neighborhood aggregation slightly outperformed GAT's learned attention mechanism on this benchmark. Both models classify well above the ~2.5% random-guess baseline for 40 classes.

## Repository Structure

```
Graph Neural Networks for Node Classification on OGBN-Arxiv/
├── Notebooks/
│   ├── 01_tensor_fundamentals.ipynb      Tensor operations and GPU basics
│   ├── 02_graph_analysis.ipynb           Graph representation, degree/density/component analysis
│   ├── 03_data_preparation.ipynb         Feature loading, official split, normalization
│   ├── 04_gnn_models.ipynb               GCN and GAT model definitions
│   ├── 05_training.ipynb                 Training loops, loss curves, hyperparameter sweep
│   ├── 06_evaluation.ipynb               Accuracy / Precision / Recall / F1 comparison
│   └── 07_explainability.ipynb           t-SNE embeddings, GAT attention weight analysis
│
├── Dashboard/
│   ├── app.py                            Streamlit dashboard application
│   ├── dashboard_data.pkl                Precomputed results used by the dashboard
│   ├── embedding_labels.npy
│   ├── embeddings_2d.npy
│   ├── graph_stats.json
│   └── model_metrics.csv
│
├── Models/
│   ├── gcn_trained.pt                    Trained GCN weights
│   └── gat_trained.pt                    Trained GAT weights
│
├── Report/
│   └── CCS4354_Technical_Report.pdf
│
├── Slides/
│   └── CCS4354_Viva_Presentation.pptx
│
├── .gitignore
└── README.md
```

## Architecture Summary

| Hyperparameter | GCN | GAT |
|---|---|---|
| Aggregation | Fixed, degree-normalized average | Learned attention weights per neighbor |
| Layers | 3 | 2 |
| Hidden dimension | 256 | 32 per head (256 effective) |
| Attention heads | — | 8 |
| Activation | ReLU | ELU |
| Dropout | 0.5 | 0.5 |
| Learning rate | 0.01 | 0.005 |
| Optimizer | Adam (weight_decay=5e-4) | Adam (weight_decay=5e-4) |
| Epochs | 200 | 200 |

## Setup & Running the Notebooks

```bash
pip install torch torch_geometric ogb pandas numpy matplotlib scikit-learn networkx
```

Notebooks were developed and run on **Google Colab** with a free GPU runtime. Open any notebook in `Notebooks/` in Colab or Jupyter and run top to bottom.

## Running the Dashboard

```bash
cd Dashboard
pip install streamlit pandas numpy matplotlib scikit-learn
python -m streamlit run app.py
```
This opens an interactive dashboard at `http://localhost:8501` showing:
- Live graph statistics (nodes, edges, density)
- Model performance comparison (table + chart)
- A node-by-node prediction explorer
- t-SNE embedding visualization

## Explainability

- **t-SNE embedding visualization** — GCN's learned node representations, reduced to 2D, show visually distinct clusters by subject category, confirming the model groups semantically similar papers together.
- **GAT attention weight analysis** — per-edge attention weights show the model does not treat all citation neighbors equally, learning to weight more informative neighbors more heavily.

## Future Work

- Deeper hyperparameter search across layer count, hidden dimension, and dropout
- A third architecture (GIN) for additional coursework marks
- GNNExplainer-based subgraph-level explanations
- Larger-scale, systematic attention analysis across many nodes

## Team — Individual Contributions

| Member | Contribution |
|---|---|
| Member 1 | Tensor fundamentals notebook, dataset setup, presentation coordination |
| Member 2 | Graph representation & analysis, data preparation, split justification |
| Member 3 | GNN architecture design (GCN & GAT), training loop, hyperparameter tuning |
| Member 4 | Model evaluation, explainability analysis, Streamlit dashboard, technical report compilation |

## References

- Open Graph Benchmark — OGBN-Arxiv Dataset: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Veličković, P. et al. (2018). *Graph Attention Networks*. ICLR.
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io
- Hu, W. et al. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*. NeurIPS.

---

*Coursework submission for CCS4354 — Tensors and Graphs, SLTC Research University, 2026.*
