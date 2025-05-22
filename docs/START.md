# GrowForever: 图结构人工智能的交互应用

GrowForever 是一个基于 GPT 进行辅助文本生成，以 GNN 为核心的结构化编码系统，通过 Text → Graph → Text 的循环机制实现专业知识的结构化编程与交互形成，并选用实际物理论文和科普论文为训练基础，自选各类实体和关系进行规范化表达。

------

## ✅ 环境搭建与开发流程（GrowNet 项目）

### 📁 项目结构（GrowNet）

```
GrowNet/
├── README.md                     # 项目介绍（你已经在写）
├── environment.yml              # Conda 环境定义
├── grownet-start.bat            # 一键启动脚本
├── .gitignore                   # 忽略文件配置
├── main.py                      # 主入口文件（可启动 TGT 流程）
├── config/
│   └── config.yaml              # 配置文件（模型路径、超参等）
├── data/
│   ├── raw/                     # 原始文本语料
│   ├── processed/               # 结构化后的图数据
│   └── external/                # 预训练模型、NER模型等
├── src/
│   ├── __init__.py
│   ├── gpt_module.py            # GPT 文本生成模块
│   ├── ner_extraction.py        # 实体识别与关系抽取
│   ├── graph_builder.py         # 文本转图结构
│   ├── graph_transformer.py     # 图结构建模（Graphormer 等）
│   ├── contrastive_loss.py      # 对比学习损失
│   ├── graph_to_text.py         # 图转文本模块
│   └── loop_controller.py       # 循环控制逻辑
├── notebooks/
│   └── 01_TGT实验记录.ipynb
├── logs/                        # 日志输出
├── outputs/                     # 模型输出与图结构
└── scripts/
    └── run_tgt.py              # 测试脚本
```

------

## ✅ Conda 环境配置文件（environment.yml）

```yaml
name: grownet
channels:
  - nvidia
  - pytorch
  - pyg
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch=2.1
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - pyg
  - numpy
  - pandas
  - scikit-learn
  - spacy
  - jupyterlab
  - matplotlib
  - pip
  - pip:
      - transformers
      - openai
      - sentence-transformers
      - faiss-cpu
      - wandb
      - gradio
      - python-dotenv
```

------

## ✅ 使用说明

### 1. 创建/更新 Conda 环境

```bash
# 在 Miniconda Prompt 中执行
conda activate base
conda env create -f environment.yml    # 首次创建
conda env update -f environment.yml --prune  # 更新已有环境
conda activate grownet
```

### 2. 启动脚本：grownet-start.bat

```bat
@echo off
CALL "C:\Users\Mingjun Zhao\miniconda3\Scripts\activate.bat"
CALL conda activate grownet
cd /d "C:\Users\Mingjun Zhao\PycharmProjects\GrowNet"
python main.py
pause
```

------

## ✅ 主流程脚本：main.py 示例结构

```python
from src.gpt_module import generate_text
from src.ner_extraction import extract_entities_and_relations
from src.graph_builder import build_graph
from src.graph_to_text import graph_to_text

def main():
    prompt = "我想去火星生活"
    profile = {"age": 22, "gender": "male"}
    text = generate_text(prompt, profile)
    graph_data = extract_entities_and_relations(text)
    graph = build_graph(graph_data['nodes'], graph_data['edges'])
    back_text = graph_to_text(graph, explored_nodes=[])
    print(back_text)

if __name__ == "__main__":
    main()
```

------

## ✅ 已实现模块 MOCK 示例（src）

```python
# src/gpt_module.py

def generate_text(prompt, user_profile):
    return "你希望在火星生活，这需要建设殖民基地、解决能源供应、建立生态系统。"

# src/ner_extraction.py

def extract_entities_and_relations(text):
    return {
        "nodes": [
            {"id": "1", "label": "火星", "type": "地点"},
            {"id": "2", "label": "殖民基地", "type": "设施"},
            {"id": "3", "label": "能源供应", "type": "需求"}
        ],
        "edges": [
            {"from": "1", "to": "2", "label": "需要建设"},
            {"from": "2", "to": "3", "label": "依赖"}
        ]
    }

# src/graph_to_text.py

def graph_to_text(graph, explored_nodes):
    return "你可以进一步探索能源供应系统，它涉及太阳能采集和氢燃料合成。"
```

------

✅ 后续模块：Graph Transformer、对比学习、图结构可视化将逐步接入。