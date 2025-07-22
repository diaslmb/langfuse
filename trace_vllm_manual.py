import requests
from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse(
    public_key="pk-lf-ecc4dc88-5f1c-49d2-8fee-9fd119ba833d",
    secret_key="sk-lf-66c3b9c6-34cc-4aa0-92c4-e813ba67b257",
    host="https://cloud.langfuse.com"
)

# vLLM configuration
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

def call_vllm_with_tracing(prompt: str):
    # ✅ Create a trace using the correct method
    trace = langfuse.traces.create(name="vLLM Inference", user_id="user-001")

    # ✅ Create and start a span
    span = langfuse.spans.create(name="vLLM Chat Completion", trace_id=trace.id)
    langfuse.spans.update(span.id, start_time_now=True)

    try:
        # Send request to vLLM
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

        # ✅ Log generation in Langfuse
        langfuse.generations.create(
            trace_id=trace.id,
            name="chat-generation",
            prompt=str(payload["messages"]),
            completion=reply,
            model=MODEL_NAME,
        )

        # ✅ End span
        langfuse.spans.update(span.id, end_time_now=True, output=reply)

        return reply

    except Exception as e:
        langfuse.spans.update(span.id, end_time_now=True)
        langfuse.spans.log(span.id, name="vLLM Error", level="ERROR", message=str(e))
        raise

# Run
if __name__ == "__main__":
    prompt = "What is Langfuse and how can it help in LLM observability?"
    result = call_vllm_with_tracing(prompt)
    print("\nLLM Output:\n", result)
