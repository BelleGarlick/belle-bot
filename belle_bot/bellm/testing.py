import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoProcessor, AutoModelForMultimodalLM
from llama_cpp import Llama

# model can be downloaded from https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/blob/main/gemma-4-E2B_q4_0-it.gguf

model_path = "/Users/belle/Downloads/gemma-4-E2B_q4_0-it.gguf"


# Load the model
# n_ctx is the context window size
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1)

if __name__ == "__main__":
    print("Warming up engine and compiling GPU shaders...")
    # 1. Warm up BOTH ingestion AND generation by forcing text output
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Say the letter x"}],
        max_tokens=5  # Forces compilation of generation kernels
    )

    # 2. Start the actual robotic action loop
    print("Engine warm. Running real-time inference...")
    a = time.time()

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a robot."},
            {"role": "user",
             "content": "Please choose an option to perform. a: play some audio, b: move to a defined position, c: move the camera. output only a, b, or c"}
        ],
        max_tokens=1  # Strict constraint prevents unnecessary generation delay
    )
    b = time.time()

    print(f"Result: {output['choices'][0]['message']['content']}")
    print(f"Pure generation loop time: {b - a:.4f} seconds")