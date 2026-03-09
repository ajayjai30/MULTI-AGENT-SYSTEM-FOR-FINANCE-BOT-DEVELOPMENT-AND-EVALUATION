# test_evaluation.py

import json
from dotenv import load_dotenv
import os
import openai # <-- Import the main openai library

# Import your project components
from langgraph_evaluation_engine import app

# Load environment variables (like OPENROUTER_API_KEY)
load_dotenv()

# --- THE DEFINITIVE FIX: PATCHING BOTH OPENAI CLIENTS ---

# 1. Store the original classes
_original_openai_client = openai.OpenAI
_original_async_openai_client = openai.AsyncOpenAI

# 2. Create our custom client instances that point to OpenRouter
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"

sync_client = _original_openai_client(
    base_url=openrouter_base_url,
    api_key=openrouter_api_key
)
async_client = _original_async_openai_client(
    base_url=openrouter_base_url,
    api_key=openrouter_api_key
)

# 3. Replace BOTH original classes with a lambda that accepts any arguments
# but returns our pre-configured client.
openai.OpenAI = lambda *args, **kwargs: sync_client
openai.AsyncOpenAI = lambda *args, **kwargs: async_client

# --- END OF FIX ---


# Define the sample inputs to be evaluated, based on the notebook
sample_question = "Explain what a neural network is in simple terms for a beginner."
sample_answer = """
Of course! A neural network is like a computer's brain. 
It is made of interconnected nodes, similar to neurons, that process information. 
When you show it many examples, like photos of dogs, it learns to spot patterns. 
This allows it to identify dogs in new photos it has never seen before.
"""

initial_state = {
    "question": sample_question,
    "answer": sample_answer,
}

print("🚀 Starting LangGraph multi-agent evaluation...")

try:
    # Invoke the graph. It will now use the correct clients.
    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print("✅ EVALUATION COMPLETE".center(50))
    print("="*50)

    # Print the final state which contains all the results
    print(json.dumps(final_state, indent=2, default=str))

    print("\n" + "="*50)
    print("📊 Check LangSmith for a detailed visual trace of the run.".center(50))
    print("="*50)

finally:
    # 4. Restore the original classes after the script is done
    openai.OpenAI = _original_openai_client
    openai.AsyncOpenAI = _original_async_openai_client
    print("\n(Restored original OpenAI clients)")