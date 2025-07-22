import os
from langfuse.openai import openai # This is the key: import from langfuse.openai
from langfuse import get_client # For explicit flushing

# --- 1. Set Langfuse API Keys (using your provided keys) ---
# For demonstration, we'll set them directly here.
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-ecc4dc88-5f1c-49d2-8fee-9fd119ba833d"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-66c3b9c6-34cc-4aa0-92c4-e813ba67b257"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # Default cloud host

# --- 2. Configure OpenAI Client to Point to Your vLLM Server ---
vllm_api_base = "http://localhost:8000/v1"

# IMPORTANT: For vLLM when it does NOT require an API key,
# you still need to pass something to the OpenAI client.
# "EMPTY" or "sk-no-key-required" or any non-empty string usually works.
# The server side must truly NOT be checking it.
# If the curl test in Step 1 passed, this is the correct setup.
dummy_api_key = "sk-no-key-required" # A common dummy value

# Initialize the OpenAI client directly, passing base_url and api_key
# This bypasses potential env variable conflicts.
# Note: When importing from `langfuse.openai`, it automatically instruments this client.
client = openai.OpenAI(
    base_url=vllm_api_base,
    api_key=dummy_api_key, # Pass the dummy key here
)

# --- 3. Make an LLM Call using the Langfuse-wrapped OpenAI client ---
vllm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Attempting to connect to vLLM at: {vllm_api_base}")
print(f"Using model: {vllm_model_name}")

try:
    completion = client.chat.completions.create( # Use 'client' instance now
        model=vllm_model_name,
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": "What is the capital of Kazakhstan?"},
        ],
        temperature=0.7,
        max_tokens=50,
        extra_body={
            "langfuse_name": "vllm-kazakhstan-capital-query",
            "langfuse_user_id": "user-kazakh-explorer",
            "langfuse_tags": ["vllm", "mistral", "geography"],
        }
    )

    print("\n--- LLM Response ---")
    print(completion.choices[0].message.content)
    print("\n--- Token Usage ---")
    print(f"Prompt Tokens: {completion.usage.prompt_tokens}")
    print(f"Completion Tokens: {completion.usage.completion_tokens}")
    print(f"Total Tokens: {completion.usage.total_tokens}")

except Exception as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("\nTroubleshooting Tips:")
    print("1. Ensure your vLLM server is running and accessible at 'http://localhost:8000'.")
    print("2. Verify your vLLM server is *not* requiring an API key (check `VLLM_API_KEY` env var on server side, and `curl` test).")
    print("3. Check your network connectivity and firewall settings.")
    print("4. Confirm your Langfuse API keys and host are correct.")

# --- 4. Flush the Langfuse Client (Important for short-lived scripts) ---
print("\nFlushing Langfuse client...")
langfuse_client = get_client()
langfuse_client.flush()
print("Langfuse client flushed. Check your Langfuse dashboard for traces.")
