from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_text(prompt: str, user_profile: dict) -> str:
    context = f"Prompt: {prompt}\nUser: {user_profile}\nThought:"
    inputs = tokenizer.encode(context, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
