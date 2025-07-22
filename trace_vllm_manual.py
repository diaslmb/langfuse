import os
from langfuse.openai import openai # Crucial: Import 'openai' from 'langfuse.openai'
from langfuse import get_client # For explicit flushing

# --- 1. Set Langfuse API Keys (using your provided keys) ---
# It's best practice to load these from environment variables or a secure config.
# For demonstration, we'll set them directly here.
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-ecc4dc88-5f1c-49d2-8fee-9fd119ba833d"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-66c3b9c6-34cc-4aa0-92c4-e813ba67b257"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # Default cloud host

# --- 2. Configure OpenAI Client to Point to Your vLLM Server ---
# Assuming your vLLM server is running locally on port 8000 and exposes an OpenAI-compatible API.
# If your vLLM server is at a different address or port, adjust this URL.
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

# vLLM's OpenAI API usually doesn't require an actual API key.
# You can set it to "EMPTY" or any non-empty string.
os.environ["OPENAI_API_KEY"] = "EMPTY" 

# --- 3. Make an LLM Call using the Langfuse-wrapped OpenAI client ---
# The model name should match how vLLM exposes 'mistralai/Mistral-7B-Instruct-v0.2'.
# Often, it's just the model ID you passed to vLLM's --model argument.
vllm_model_name = "mistralai/Mistral-7B-Instruct-v0.2" # Use the exact model name vLLM is serving

print(f"Attempting to connect to vLLM at: {os.environ['OPENAI_API_BASE']}")
print(f"Using model: {vllm_model_name}")

try:
    completion = openai.chat.completions.create(
        model=vllm_model_name,
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": "What is the capital of Kazakhstan?"},
        ],
        temperature=0.7,
        max_tokens=50,
        # Optional: Add custom metadata that will appear in Langfuse
        # These are passed via the 'extra_body' parameter for Langfuse's OpenAI wrapper.
        extra_body={
            "langfuse_name": "vllm-kazakhstan-capital-query", # A readable name for this trace
            "langfuse_user_id": "user-kazakh-explorer", # An identifier for the user
            "langfuse_tags": ["vllm", "mistral", "geography"], # Custom tags
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
    print("   (e.g., `python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --host 0.0.0.0`)")
    print("2. Verify that vLLM is serving the 'mistralai/Mistral-7B-Instruct-v0.2' model correctly.")
    print("3. Check your network connectivity and firewall settings.")
    print("4. Confirm your Langfuse API keys and host are correct.")


# --- 4. Flush the Langfuse Client (Important for short-lived scripts) ---
# This ensures that all traces and observations are sent to the Langfuse server.
# For long-running applications (like web servers), Langfuse handles flushing automatically
# or you can configure a periodic flush.
print("\nFlushing Langfuse client...")
langfuse_client = get_client()
langfuse_client.flush()
print("Langfuse client flushed. Check your Langfuse dashboard for traces.")
