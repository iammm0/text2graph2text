from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

def visualize_embeddings(embeddings: torch.Tensor, labels=None):
    num_nodes = embeddings.shape[0]

    if num_nodes < 2:
        print("⚠️ 无法可视化：嵌入向量数量太少（至少需要两个节点）")
        return

    # 安全设置 perplexity（不能大于样本数且必须 > 0）
    perplexity = max(2, min(30, num_nodes - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity)

    reduced = tsne.fit_transform(embeddings.detach().numpy())

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='skyblue', edgecolors='k')

    if labels and len(labels) == num_nodes:
        for i, txt in enumerate(labels):
            plt.annotate(txt, (reduced[i, 0], reduced[i, 1]))

    plt.title("Graph Node Embedding Visualization (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()
