import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Graph Intelligence Dashboard", layout="wide")

# ---- Load precomputed data ----
@st.cache_data
def load_data():
    with open('dashboard_data.pkl', 'rb') as f:
        return pickle.load(f)

data = load_data()

# ---- Title ----
st.title("📊 Graph Intelligence Dashboard — OGBN-Arxiv")
st.markdown("Interactive dashboard for exploring GNN model performance and graph structure.")

# ---- Section 1: Graph Statistics ----
st.header("1. Graph Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Number of Nodes", f"{data['num_nodes']:,}")
col2.metric("Number of Edges", f"{data['num_edges']:,}")
col3.metric("Graph Density", f"{data['density']:.6f}")

# ---- Section 2: Model Performance ----
st.header("2. Model Performance Metrics")
metrics_df = pd.DataFrame(data['metrics_df'])
st.dataframe(metrics_df)

# Bar chart comparison
st.subheader("Test Set Comparison")
fig, ax = plt.subplots(figsize=(8, 4))
test_rows = metrics_df[metrics_df['Split'] == 'Test']
metrics_cols = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
x = np.arange(len(metrics_cols))
width = 0.35
for i, model_name in enumerate(test_rows['Model'].unique()):
    vals = test_rows[test_rows['Model'] == model_name][metrics_cols].values[0]
    ax.bar(x + i * width, vals, width, label=model_name)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(metrics_cols, rotation=15)
ax.legend()
st.pyplot(fig)

# ---- Section 3: Node Classification Explorer ----
st.header("3. Node Classification Results")
node_options = list(range(len(data['y_true_gcn_test'])))
selected_node = st.selectbox("Select a test node (index):", node_options[:200])  # limit dropdown size
st.write(f"**True label:** {data['y_true_gcn_test'][selected_node]}")
st.write(f"**GCN predicted label:** {data['y_pred_gcn_test'][selected_node]}")
correct = data['y_true_gcn_test'][selected_node] == data['y_pred_gcn_test'][selected_node]
st.success("✅ Correct prediction") if correct else st.error("❌ Incorrect prediction")

# ---- Section 4: Embedding Visualization ----
st.header("4. Node Embedding Visualization (t-SNE)")
fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter = ax2.scatter(
    data['embeddings_2d'][:, 0], data['embeddings_2d'][:, 1],
    c=data['embedding_labels'], cmap='tab20', s=8, alpha=0.7
)
plt.colorbar(scatter, ax=ax2, label='Subject Category')
ax2.set_title('GCN Node Embeddings')
st.pyplot(fig2)

st.markdown("---")
st.caption("Built for CCS4354 — Tensors and Graphs Coursework")
