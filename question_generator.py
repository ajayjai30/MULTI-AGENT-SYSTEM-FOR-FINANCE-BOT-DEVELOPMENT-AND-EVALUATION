# question_generator.py
import json
from llm_wrapper import get_llm

def generate_test_conversation(bot_config: dict, num_turns: int = 3) -> list:
    """
    Uses an LLM to generate a sequence of conversational questions tailored to the bot's config.
    """
    print(f"🤖 Generating a {num_turns}-turn test conversation...")
    
    # This prompt instructs the LLM to act as a QA expert and create a conversation
    generator_prompt = f"""
You are a Quality Assurance (QA) expert creating a test script for a conversational AI.
Your task is to generate a sequence of exactly {num_turns} user questions that represent a natural, back-and-forth conversation.

**The Chatbot's Configuration:**
- Archetype: {bot_config.get("archetype")}
- Domain Knowledge: {", ".join(bot_config.get("domain_knowledge", []))}

**Instructions:**
Create a conversation that starts with a broad topic in the bot's domain and becomes more specific with each follow-up question. The questions should test the chatbot's ability to remember context.

Return your response ONLY as a valid JSON list of strings. For example:
[
    "What is a good way to start saving money?",
    "That makes sense. For that first step, how much should I aim to save each month?",
    "Okay, based on that amount, can you suggest some ways to cut my daily expenses?"
]
"""
    
    llm = get_llm()
    
    try:
        response_str = llm(system_prompt_content=generator_prompt)
        
        # Clean up the response in case the model wraps it in markdown
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0]

        generated_questions = json.loads(response_str)
        if isinstance(generated_questions, list) and len(generated_questions) == num_turns:
            print("✅ Successfully generated test conversation.")
            return generated_questions
        else:
            raise ValueError("Generated content is not a list of the correct length.")

    except (json.JSONDecodeError, ValueError, Exception) as e:
        print(f"Error generating questions: {e}. Using a default conversation as a fallback.")
        # Provide a default list in case the generation fails
        return [
            "What is the 50/30/20 budgeting rule?",
            "Is it a good strategy for someone with an inconsistent income?",
            "What is a good alternative for me?"
        ]