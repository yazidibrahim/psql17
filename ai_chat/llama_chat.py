from llama_cpp import Llama

MODEL_PATH = "/home/cybrosys/Downloads/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

def get_response(input_text):
    output = llm(f"User: {input_text}\nAssistant:", max_tokens=100)
    return output["choices"][0]["text"].strip()