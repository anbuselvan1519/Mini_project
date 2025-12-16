from llama_cpp import Llama
import os

# Load once at startup
llm = Llama(
    model_path=r"C:\Users\admin\OneDrive\Documents\projects\llama.cpp\models\llama-3.2-1b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=max(1, os.cpu_count() // 2),
    n_gpu_layers=20  # set 0 if CPU-only
)

def chat_with_ai(prompt: str) -> str:
    """Ask LLaMA a question and return its answer."""
    response = llm(
        f"You are an agricultural assistant AI. Answer clearly.\n\nUser: {prompt}\nAI:",
        max_tokens=256,
        stop=["\nUser:", "\nAI:"]
    )
    return response['choices'][0]['text'].strip()
