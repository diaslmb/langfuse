import requests
from langfuse import Langfuse

# === Langfuse client ===
langfuse = Langfuse(
    public_key="pk-lf-ecc4dc88-5f1c-49d2-8fee-9fd119ba833d",
    secret_key="sk-lf-66c3b9c6-34cc-4aa0-92c4-e813ba67b257",
    host="https://cloud.langfuse.com"
)

# === vLLM endpoint config ===
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def call_vllm_with_tracing(prompt: str):
    # ✅ Create trace directly
    trace = langfuse.create_trace(name="vLLM Inference", user_id="user-001")
    span = trace.create_span(name="vLLM Chat Completion")
    span.start()

    try:
        # Call vLLM
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        response = requests.post(VLLM_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        # Log generation
        langfuse.create_generation(
            trace_id=trace.id,
            name="chat-generation",
            prompt=str(payload["messages"]),
            completion=reply,
            model=MODEL_NAME,
        )

        span.end(output=reply)
        return reply

    except Exception as e:
        span.log_error(name="vLLM Error", error=e)
        span.end()
        raise

# === Run test
if __name__ == "__main__":
    prompt = "Explain Langfuse observability for custom LLMs."
    result = call_vllm_with_tracing(prompt)
    print("\nLLM Output:\n", result)
