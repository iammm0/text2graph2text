"""GPT 文本生成模块"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import CONFIG

# 根据配置加载预训练模型与分词器
tokenizer = AutoTokenizer.from_pretrained(CONFIG["gpt"]["model_name"])
model = AutoModelForCausalLM.from_pretrained(CONFIG["gpt"]["model_name"])


def generate_text(prompt: str, user_profile: dict) -> str:
    """根据提示词和用户画像生成文本"""

    # 构造上下文，融入用户信息
    context = f"Prompt: {prompt}\nUser: {user_profile}\nThought:"
    inputs = tokenizer.encode(context, return_tensors="pt")

    # 调用语言模型生成文本
    outputs = model.generate(
        inputs,
        max_new_tokens=CONFIG["gpt"]["max_new_tokens"],
        temperature=CONFIG["gpt"].get("temperature", 1.0),
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
