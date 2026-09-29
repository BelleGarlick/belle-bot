import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run_inference(
    base_model_name: str = "Qwen/Qwen3-0.6B",
    adapter_path: str = "./qwen-finetuned",
    prompt: str = "Who wrote Romeo and Juliet?"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Load the fine-tuned adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    print("Response:", content)

if __name__ == "__main__":
    run_inference()
