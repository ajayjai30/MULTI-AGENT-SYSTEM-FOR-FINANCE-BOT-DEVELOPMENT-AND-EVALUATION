from prompt_builder import build_prompt
from llm_wrapper import get_llm

def create_bot_chain(bot_config: dict = None, custom_prompt: str = None):
    """
    Creates a chatbot function.

    This function is flexible and can be initialized in two ways:
    1.  With a 'bot_config' dictionary: It will use the prompt_builder to create a detailed prompt.
    2.  With a 'custom_prompt' string: It will use this string directly as the system prompt.
    """
    if custom_prompt:
        prompt_text = custom_prompt
    elif bot_config:
        prompt_text = build_prompt(bot_config)
    else:
        # Error handling if neither config nor custom prompt is provided
        raise ValueError("You must provide either a 'bot_config' dictionary or a 'custom_prompt' string.")

    llm = get_llm()

    def chat_function(user_input, history):
        response = llm(
            system_prompt_content=prompt_text, 
            user_content=user_input,
            history=history
        )
        return response
    
    return chat_function