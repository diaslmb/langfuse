import requests
from langfuse import Langfuse

# === Initialize Langfuse client ===
langfuse = Langfuse(
    public_key="pk-lf-ecc4dc88-5f1c-49d2-8fee-9fd119ba833d",
    secret_key="sk-lf-66c3b9c6-34cc-4aa0-92c4-e813ba67b257",
    host="https://cloud.langfuse.com"
)

# === vLLM Endpoint ===
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def call_vllm_with_tracing(prompt: str):
    # 1. Create trace
    trace = langfuse.trace({
        "name": "vLLM Inference",
        "user_id": "user-001"
    })

    # 2. Create span
    span = langfuse.span({
        "name": "vLLM Chat Completion",
        "trace_id": trace.id
    })
    span.start()

    try:
        # 3. Make vLLM request
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        response = requests.post(VLLM_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        reply = result["choices"][0]["message"]["content"]

        # 4. Log generation
        langfuse.generation({
            "trace_id": trace.id,
            "name": "chat-generation",
            "prompt": str(payload["messages"]),
            "completion": reply,
            "model": MODEL_NAME
        })

        # 5. End span
        span.end(output=reply)
        return reply

    except Exception as e:
        span.log_error(name="vLLM Error", error=e)
        span.end()
        raise

# === Run test
if __name__ == "__main__":
    prompt = "Explain what Langfuse does in an LLM production stack."
    result = call_vllm_with_tracing(prompt)
    print("\nLLM Output:\n", result)
