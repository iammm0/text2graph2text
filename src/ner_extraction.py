import spacy
from langdetect import detect
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# 延迟加载（只加载一次）
_spacy_models = {}


def get_nlp(text):
    lang = detect(text)
    if lang.startswith("zh"):
        model_name = "zh_core_web_sm"
    else:
        model_name = "en_core_web_sm"

    if model_name not in _spacy_models:
        _spacy_models[model_name] = spacy.load(model_name)

    return _spacy_models[model_name]


import os
import json

client = OpenAI(api_key="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")

def extract_entities_and_relations_gpt(text):
    system_prompt = (
        "You are a knowledge structure extraction agent. "
        "Given a paragraph of English text, extract key conceptual entities and their semantic relationships. "
        "Return the result as a JSON with two fields: 'nodes' and 'edges'. "
        "Each node should have id, label, and type. Each edge should have from, to, and label."
    )

    user_prompt = f"""Text:
{text}

Please output a valid JSON in the following format:
{{
  "nodes": [
    {{ "id": "1", "label": "ambition", "type": "emotion" }},
    {{ "id": "2", "label": "self-doubt", "type": "emotion" }},
    ...
  ],
  "edges": [
    {{ "from": "1", "to": "2", "label": "conflicts_with" }},
    ...
  ]
}}"""

    try:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )

        content = response.choices[0].message.content

        # 安全解析 JSON（防止 GPT 出锅）
        json_start = content.find('{')
        json_data = json.loads(content[json_start:])
        return json_data

    except Exception as e:
        print(f"[GPT抽结构模块] 解析失败：{e}")
        return {"nodes": [], "edges": []}



def extract_entities_and_relations(text: str) -> dict:
    nlp = get_nlp(text)
    doc = nlp(text)
    nodes = []
    edges = []

    for ent_id, ent in enumerate(doc.ents):
        nodes.append({
            "id": str(ent_id),
            "label": ent.text,
            "type": ent.label_
        })

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append({
                "from": nodes[i]["id"],
                "to": nodes[j]["id"],
                "label": "共现"
            })

    return {
        "nodes": nodes,
        "edges": edges
    }
