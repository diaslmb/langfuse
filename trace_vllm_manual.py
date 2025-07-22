import requests
from langfuse import Langfuse

# Langfuse credentials
langfuse = Langfuse(
    public_key="pk-lf-ecc4dc88-5f1c-49d2-8fee-9fd119ba833d",
    secret_key="sk-lf-66c3b9c6-34cc-4aa0-92c4-e813ba67b257",
    host="https://cloud.langfuse.com"
)

# vLLM endpoint (chat completion format)
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

def call_vllm_with_tracing(prompt: str):
    # Start a trace and span
    trace = langfuse.trace(name="vLLM Inference", user_id="user-001")
    span = trace.span(name="vLLM Chat Completion")
    span.start()

    try:
        # Compose OpenAI-compatible vLLM request
        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",  # adapt to your model
            "messages": [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        # Send request to local vLLM server
        response = requests.post(VLLM_API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        # Log the generation in Langfuse
        langfuse.generation(
            trace_id=trace.id,
            name="chat-generation",
            prompt=str(payload["messages"]),
            completion=reply,
            model="mistralai/Mistral-7B-Instruct-v0.2",
        )

        # End the span with output
        span.end(output=reply)

        return reply

    except Exception as e:
        span.log_error(name="vLLM Error", error=e)
        span.end()
        raise

# Test run
if __name__ == "__main__":
    prompt = "What is Langfuse and how can it help trace LLM apps?"
    output = call_vllm_with_tracing(prompt)
    print("\nLLM Output:\n", output)
