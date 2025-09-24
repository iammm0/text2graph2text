## 内部接口（Pipeline 组件）

### 1. 实体 & 关系抽取

**POST** `/internal/extract_entities_relations`

- **输入**：

```json
{
  "text": "Alice lives in Paris. Bob works at Google."
}
```

- **输出**：

```json
{
  "entities": [
    {"id": "e1", "label": "Alice", "type": "Person"},
    {"id": "e2", "label": "Paris", "type": "Location"},
    {"id": "e3", "label": "Bob", "type": "Person"},
    {"id": "e4", "label": "Google", "type": "Organization"}
  ],
  "relations": [
    {"source": "e1", "target": "e2", "relation": "lives_in"},
    {"source": "e3", "target": "e4", "relation": "works_at"}
  ]
}
```

------

### 2. 图构建

**POST** `/internal/build_graph`

- **输入**：实体 + 关系
- **输出**：标准化图（JSON/Neo4j 可导入格式）

------

### 3. 图编码（Graph Encoder）

**POST** `/internal/encode_graph`

- **输入**：图结构
- **输出**：

```json
{
  "graph_embedding": [0.123, -0.456, ...],
  "node_embeddings": {
    "e1": [0.1, 0.2, ...],
    "e2": [0.3, 0.4, ...]
  }
}
```

------

### 4. 图嵌入（Graph Embeddings for Prompt/Matching）

**POST** `/internal/graph_embeddings`

- **输入**：图结构 / 节点集合
- **输出**：embedding 向量，供匹配或检索使用

------

### 5. 图-文本匹配

**POST** `/internal/graph_text_match`

- **输入**：图 embedding + 文本 embedding
- **输出**：相似度分数 / 是否匹配

------

### 6. 对比学习训练

**POST** `/internal/contrastive_train`

- **输入**：训练数据（图-文本正负样本对）
- **输出**：训练日志、更新的 embedding 模型权重

------

### 7. 文本生成测试

**POST** `/internal/test_generator`

- **输入**：prompt
- **输出**：候选文本，用于调试和循环验证

------

