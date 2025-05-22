import torch
from torch_geometric.data import DataLoader

from src.contrastive_loss import nt_xent_loss
from src.gpt_module import generate_text
from src.graph_builder import build_graph
from src.graph_embedding_visualizer import visualize_embeddings
from src.graph_to_text import graph_to_text
from src.graph_transformer import GraphEncoder
from src.graph_visualizer import visualize_graph
from src.ner_extraction import extract_entities_and_relations_gpt
from src.train_contrastive import train_contrastive_epoch


def main():
    print("🎉 GrowNet: 图结构智能系统启动")

    # === 1. 用户输入 Prompt 和配置 ===
    user_prompt = "Describe the mental patterns of a person who constantly oscillates between ambition and self-doubt in the context of modern life."
    user_profile = {
        "age": 22,
        "gender": "male",
        "interests": ["space", "technology", "adventure"]
    }

    # === 2. 生成长文本 ===
    print("\n🧠 Step 1: 文本生成")
    long_text = generate_text(user_prompt, user_profile)
    print("📄 生成文本:\n", long_text)

    # === 3. 实体识别 + 关系抽取 ===
    print("\n🔍 Step 2: 实体识别与关系抽取")
    graph_data = extract_entities_and_relations_gpt(long_text)
    print("🧩 图结构数据:", graph_data)

    # === 4. 构建图 ===
    print("\n🕸 Step 3: 构建图")
    graph = build_graph(graph_data['nodes'], graph_data['edges'])
    print(f"✅ 成功构建图：{graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")

    # === 5. 可视化图结构 ===
    visualize_graph(graph, output_file="graph_output.html")

    # === 6. 准备图神经模型输入 ===
    print("\n🔬 Step 4: 图结构编码")
    # === 重新编号：把字符串 ID → int 索引 ===
    node_id_map = {node_id: idx for idx, node_id in enumerate(graph.nodes)}

    # === 构建 one-hot 特征向量（可后期替换为语义 embedding） ===
    num_nodes = len(node_id_map)
    x = torch.eye(num_nodes)

    # === 构建 edge_index，用 int 索引 ===
    edge_index = torch.tensor([
        [node_id_map[src] for src, tgt in graph.edges],
        [node_id_map[tgt] for src, tgt in graph.edges]
    ], dtype=torch.long)

    # === 构建图数据对象 ===
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)

    # === 7. 运行图编码器 ===
    model = GraphEncoder(in_channels=num_nodes, hidden_channels=32, out_channels=16)
    embeddings = model(data.x, data.edge_index)
    print("🧠 得到节点语义嵌入，shape:", embeddings.shape)

    # === 8. 图嵌入可视化 ===
    visualize_embeddings(embeddings, labels=[graph.nodes[n]["label"] for n in graph.nodes])

    # === 9. 示例对比学习损失计算（模拟一对） ===
    print("\n🎯 Step 5: 对比学习")
    mid = embeddings.shape[0] // 2
    if mid > 0:
        z1 = embeddings[:mid]
        z2 = embeddings[mid:mid * 2]
        loss = nt_xent_loss(z1, z2)
        print("🧠 对比学习 NT-Xent Loss:", loss.item())
    else:
        print("⚠️ 节点太少，跳过对比")

    # === 10. 从图结构生成新的文本提示 ===
    print("\n📝 Step 6: 图转文本探索")
    text_prompt = graph_to_text(graph, embeddings, target_node_id="0")
    print("📜 生成的新探索提示：", text_prompt)

    # === 11. 训练 GNN 图嵌入模型 ===
    print("\n🚀 Step 7: 图嵌入对比训练")
    loader = DataLoader([data, data], batch_size=2)  # 简化 demo：双图视为一对
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(10):
        loss = train_contrastive_epoch(model, loader, optimizer)
        print(f"Epoch {epoch + 1}: contrastive loss = {loss:.4f}")

if __name__ == "__main__":
    main()
