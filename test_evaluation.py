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


from bot_agent import create_bot_chain
from question_generator import generate_test_conversation

bot_config = {
    "archetype": "The Advisor",
    "communication_style": "Empathic Deliberative",
    "voice_tone": "Advisory",
    "behavioral_traits": ["Principled", "Adaptive"],
    "empathy_level": "4: Supportive (High Emotional Intelligence)",
    "domain_knowledge": ["Budgeting", "Goal Setting"],
    "logic_traits": ["Prompt Engineering", "Chain of Thought"],
    "knowledge_traits": ["Custom Facts", "Reference Materials"],
}

print("🚀 Starting Bot Evaluation Test...")

try:
    bot = create_bot_chain(bot_config)
    questions = generate_test_conversation(bot_config, num_turns=3)

    conversation_history = []

    for i, user_input in enumerate(questions):
        print(f"\n--- Turn {i+1} ---")
        print(f"👤 Question: {user_input}")

        response = bot(user_input, conversation_history)
        print(f"🤖 Answer:\n{response.strip()}\n")

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        print(f"\n🧪 Evaluating Turn {i+1}...")
        initial_state = {
            "question": user_input,
            "answer": response,
        }

        # Invoke the graph. It will now use the correct clients.
        final_state = app.invoke(initial_state)
        print("\n✅ Turn Evaluation Complete.")

    print("\n" + "="*50)
    print("✅ ALL EVALUATIONS COMPLETE".center(50))
    print("="*50)

finally:
    # 4. Restore the original classes after the script is done
    openai.OpenAI = _original_openai_client
    openai.AsyncOpenAI = _original_async_openai_client
    print("\n(Restored original OpenAI clients)")