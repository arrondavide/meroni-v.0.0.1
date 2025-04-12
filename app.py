from llama_cpp import Llama
import gradio as gr

llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=4
)

def chat(user_input, history):
    prompt = f"""### Instruction:
Your name is Meroni. You are a calm, kind, emotionally supportive AI companion. You offer thoughtful and empathetic responses.

### Conversation:
{format_history(history)}
User: {user_input}
Meroni:"""

    output = llm(prompt, max_tokens=200, stop=["User:", "Meroni:"], echo=False)
    reply = output['choices'][0]['text'].strip()
    history.append((user_input, reply))
    return history, history

def format_history(history):
    return '\n'.join([f"User: {u}\nMeroni: {a}" for u, a in history])

gr.ChatInterface(
    fn=chat,
    title="🌿 Meroni — Calm Companion",
    examples=["I feel anxious.", "What should I do when I overthink?"]
).launch()
