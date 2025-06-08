import torch
from torch_geometric.data import Data, DataLoader

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
    print("🎉 NeuroWeave 启动：文本-图谱智能生成系统")

    # ========== Step 1: 用户输入 ==========
    user_prompt = "Describe the mental patterns of a person who constantly oscillates between ambition and self-doubt in the context of modern life."
    user_profile = {
        "age": 22,
        "gender": "male",
        "interests": ["space", "technology", "adventure"]
    }

    # ========== Step 2: 文本生成 ==========
    print("\n🧠 Step 1: 文本生成")
    long_text = generate_text(user_prompt, user_profile)
    print("📄 生成文本:\n", long_text)

    # ========== Step 3: 实体与关系抽取 ==========
    print("\n🔍 Step 2: 实体识别与关系抽取")
    graph_data = extract_entities_and_relations_gpt(long_text)
    print("🧩 图结构数据:", graph_data)

    # ========== Step 4: 构建图 ==========
    print("\n🕸 Step 3: 构建图")
    graph = build_graph(graph_data["nodes"], graph_data["edges"])
    print(f"✅ 成功构建图：{graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
    visualize_graph(graph, output_file="graph_output.html")

    # ========== Step 5: 图结构编码 ==========
    print("\n🔬 Step 4: 图结构编码")
    node_id_map = {node_id: idx for idx, node_id in enumerate(graph.nodes)}
    num_nodes = len(node_id_map)
    x = torch.eye(num_nodes)

    edge_index = torch.tensor([
        [node_id_map[src] for src, tgt in graph.edges],
        [node_id_map[tgt] for src, tgt in graph.edges]
    ], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    model = GraphEncoder(in_channels=num_nodes, hidden_channels=32, out_channels=16)
    embeddings = model(data.x, data.edge_index)
    print("🧠 得到节点语义嵌入，shape:", embeddings.shape)

    if embeddings.shape[0] >= 2:
        visualize_embeddings(embeddings, labels=[graph.nodes[n]["label"] for n in graph.nodes])
    else:
        print("⚠️ 可视化跳过：节点数不足")

    # ========== Step 6: 对比学习 ==========
    print("\n🎯 Step 5: 对比学习")
    mid = embeddings.shape[0] // 2
    if mid > 0:
        z1, z2 = embeddings[:mid], embeddings[mid:mid*2]
        loss = nt_xent_loss(z1, z2)
        print("🧠 对比学习 NT-Xent Loss:", loss.item())
    else:
        print("⚠️ 节点太少，跳过对比损失")

    # ========== Step 7: 图转文本探索 ==========
    print("\n📝 Step 6: 图转文本探索")
    try:
        text_prompt = graph_to_text(graph, embeddings, target_node_id="0")
        print("📜 生成的新探索提示：", text_prompt)
    except Exception as e:
        print("⚠️ 图转文本失败：", e)

    # ========== Step 8: 图嵌入训练 ==========
    print("\n🚀 Step 7: 图嵌入对比训练")
    if num_nodes >= 2:
        loader = DataLoader([data, data], batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        for epoch in range(10):
            epoch_loss = train_contrastive_epoch(model, loader, optimizer)
            print(f"Epoch {epoch+1}: contrastive loss = {epoch_loss:.4f}")
    else:
        print("⚠️ 节点过少，跳过训练流程")

    print("\n✅ 所有流程完成，NeuroWeave 完成一次循环")

if __name__ == "__main__":
    main()
