from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Union
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    GenerationConfig
)

@dataclass
class GPTConfig:
    model_name: str = "Qwen/Qwen2-7B-Instruct"  # 示例：中文友好
    max_new_tokens: int = 1024                  # 长文可加大；注意总上下文别超模型上限
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = None
    repetition_penalty: float = 1.05
    do_sample: bool = True
    # 性能与显存
    device_map: Union[str, Dict] = "auto"
    torch_dtype: Optional[str] = "bfloat16"     # 可设为 "float16" / None
    quantization: Optional[str] = None          # "4bit" | "8bit" | None
    trust_remote_code: bool = True              # 针对 Qwen 等自定义代码模型
    # 生成细节
    eos_token_ids: Optional[List[int]] = None   # 额外停用 token（可留空）
    stop_words: Optional[List[str]] = None      # 文本停用串（中文里很实用）
    # 流式输出
    stream: bool = False

class GPTGenerator:
    def __init__(self, cfg: GPTConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=True
        )

        load_kwargs = {
            "trust_remote_code": cfg.trust_remote_code,
            "device_map": cfg.device_map,
        }

        if cfg.quantization == "4bit":
            # 需要安装 bitsandbytes，CUDA 环境
            load_kwargs.update(dict(load_in_4bit=True))
        elif cfg.quantization == "8bit":
            load_kwargs.update(dict(load_in_8bit=True))
        else:
            if cfg.torch_dtype in {"bfloat16", "float16"}:
                load_kwargs["torch_dtype"] = getattr(torch, cfg.torch_dtype)

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kwargs)

        # pad/eos 修正（有些中文模型 pad_token 为空）
        if self.tokenizer.pad_token_id is None:
            # 通常将 pad_token 对齐到 eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 预设一个通用的 GenerationConfig
        self.gen_config = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
        )
        if cfg.top_k is not None:
            self.gen_config.top_k = cfg.top_k
        if cfg.eos_token_ids:
            self.gen_config.eos_token_id = cfg.eos_token_ids

    # -------- Prompt 构造：优先使用 chat 模板 ----------
    def build_inputs(self, prompt: str, user_profile: Dict) -> Dict[str, torch.Tensor]:
        """
        1) 若模型带 chat 模板（如 Qwen/Yi/Llama3-Instruct），使用 tokenizer.apply_chat_template
        2) 否则退化为中文指令模板
        """
        messages = [
            {"role": "system", "content": "你是一个擅长中文长文本写作与推理的助手。"},
            {"role": "user", "content": self._cn_prompt(prompt, user_profile)}
        ]

        try:
            # 部分模型需要 add_generation_prompt=True 才会加 assistant 起始标记
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt")
        except Exception:
            # 回退到朴素模板
            fallback = (
                "【角色】中文写作与推理助手\n"
                "【用户画像】\n"
                f"{json.dumps(user_profile, ensure_ascii=False, indent=2)}\n"
                "【任务】\n"
                f"{prompt}\n"
                "【写作要求】语言自然、结构清晰、论证充分，必要时先给出提纲再展开。\n"
                "【开始输出】"
            )
            inputs = self.tokenizer(fallback, return_tensors="pt")

        # 自动移到模型所在设备
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @staticmethod
    def _cn_prompt(prompt: str, user_profile: Dict) -> str:
        return (
            "请根据以下用户画像与任务生成中文长文本：\n"
            f"用户画像：{json.dumps(user_profile, ensure_ascii=False)}\n"
            f"任务：{prompt}\n"
            "要求：行文连贯、信息密度高、适度引用事实或例子；先给出清晰结构或提纲，再逐节展开。"
        )

    # -------- 基础生成 ----------
    def generate_text(
        self,
        prompt: str,
        user_profile: Dict,
        extra_config: Optional[Dict] = None
    ) -> str:
        inputs = self.build_inputs(prompt, user_profile)
        gen_cfg = self._merge_gen_config(extra_config)

        # 停用词支持（文本级别）
        stop_words_ids = None
        if self.cfg.stop_words:
            stop_words_ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in self.cfg.stop_words]

        if self.cfg.stream:
            # 流式输出（如果你在控制台/前端需要边打边看）
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            outputs = self.model.generate(**inputs, generation_config=gen_cfg, streamer=streamer, pad_token_id=self.tokenizer.pad_token_id)
            # 同步消费流
            chunks = []
            for token in streamer:
                chunks.append(token)
            return "".join(chunks)
        else:
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_cfg,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -------- 长文本两阶段：先提纲，后扩写 ----------
    def generate_longform(
        self,
        prompt: str,
        user_profile: Dict,
        outline_sections: int = 6,
        section_tokens: int = 512
    ) -> str:
        # 1) 生成提纲
        outline_prompt = (
            f"{prompt}\n\n"
            f"请先输出一个结构清晰的提纲（约 {outline_sections} 节，每节一句话），"
            "随后不要展开正文。提纲输出格式：\n"
            "1. …\n2. …\n3. …\n"
        )
        outline_text = self.generate_text(outline_prompt, user_profile)

        # 2) 遍历提纲逐节扩写（避免一次性爆长）
        sections = self._extract_sections(outline_text, max_n=outline_sections)
        final_parts = [outline_text, "\n\n【正文展开】\n"]
        for i, sec in enumerate(sections, 1):
            section_prompt = (
                f"基于以下提纲第 {i} 节进行深入写作（{section_tokens} tokens 左右）：\n"
                f"提纲：{sec}\n"
                "要求：逻辑清晰、例证充分、自然过渡，段落成型，避免重复。"
            )
            part = self.generate_text(section_prompt, user_profile, extra_config={
                "max_new_tokens": section_tokens
            })
            final_parts.append(f"\n【第 {i} 节】{sec}\n{part}\n")

        return "\n".join(final_parts)

    # -------- 工具函数 ----------
    def _merge_gen_config(self, extra: Optional[Dict]) -> GenerationConfig:
        if not extra:
            return self.gen_config
        merged = self.gen_config.to_dict()
        merged.update(extra)
        return GenerationConfig(**merged)

    @staticmethod
    def _extract_sections(outline_text: str, max_n: int = 6) -> List[str]:
        lines = [l.strip() for l in outline_text.splitlines()]
        sections = []
        for l in lines:
            if not l:
                continue
            if l[0].isdigit() and "." in l[:4]:
                # 形如 "1. xxx"
                sec = l.split(".", 1)[1].strip()
                if sec:
                    sections.append(sec)
        if not sections:
            # 兜底：取前 max_n 行作为“提纲”
            sections = [l for l in lines if l][:max_n]
        return sections[:max_n]


# ======== 便捷方法：与你现有 CONFIG 结合 ========
def build_generator_from_CONFIG(CONFIG) -> GPTGenerator:
    cfg = GPTConfig(
        model_name=CONFIG["gpt"]["model_name"],
        max_new_tokens=CONFIG["gpt"].get("max_new_tokens", 1024),
        temperature=CONFIG["gpt"].get("temperature", 0.8),
        top_p=CONFIG["gpt"].get("top_p", 0.9),
        top_k=CONFIG["gpt"].get("top_k", None),
        repetition_penalty=CONFIG["gpt"].get("repetition_penalty", 1.05),
        device_map=CONFIG["gpt"].get("device_map", "auto"),
        torch_dtype=CONFIG["gpt"].get("torch_dtype", "bfloat16"),
        quantization=CONFIG["gpt"].get("quantization", None),
        trust_remote_code=CONFIG["gpt"].get("trust_remote_code", True),
        eos_token_ids=CONFIG["gpt"].get("eos_token_ids", None),
        stop_words=CONFIG["gpt"].get("stop_words", None),
        stream=CONFIG["gpt"].get("stream", False),
    )
    return GPTGenerator(cfg)
