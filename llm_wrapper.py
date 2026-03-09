from openai import OpenAI
import openai
import os
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, FALLBACK_MODELS
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    # FIX: The function now accepts a 'history' list as a parameter.
    def call_llm(system_prompt_content: str, user_content: str = "", history: list = []):
        
        models_to_try = [DEFAULT_MODEL] + FALLBACK_MODELS

        # FIX: The messages payload is now built with the history included.
        # The structure is: System Prompt -> Past Conversation -> New User Message
        messages = [{"role": "system", "content": system_prompt_content}] + history
        if user_content:
            messages.append({"role": "user", "content": user_content})

        for model in models_to_try:
            try:
                print(f"Attempting to use model: {model}...")
                response = client.chat.completions.create(
                    model=model,
                    temperature=DEFAULT_TEMPERATURE,
                    messages=messages
                )
                print(f"Successfully received response from {model}.")
                return response.choices[0].message.content
            
            except openai.BadRequestError as e:
                print(f"Error: The request was invalid, likely due to a bad model ID: {model}. Details: {e}")
                return f"Fatal Error: The model ID '{model}' is invalid. Please check your config.py."

            except openai.InternalServerError as e:
                if e.status_code == 503:
                    print(f"Warning: Model {model} is unavailable (503 Error). Trying next model...")
                    continue
                else:
                    return f"A server error occurred with model {model}: {e}"
            
            except Exception as e:
                print(f"An unexpected error occurred with model {model}: {e}")
                return f"An unexpected error occurred: {e}"

        return "Sorry, all available models are temporarily unavailable. Please try again later."
    
    return call_llm