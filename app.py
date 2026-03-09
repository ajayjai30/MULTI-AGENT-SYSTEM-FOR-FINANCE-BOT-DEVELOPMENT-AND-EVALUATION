import gradio as gr
from bot_agent import create_bot_chain
from langgraph_evaluation_engine import app as eval_app
import json

# Define the bot configuration
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

# Create the bot instance
bot = create_bot_chain(bot_config)

def chat_and_evaluate(user_message, history):
    # Format history for the bot:
    # history in Gradio is typically [[user_msg, bot_msg], [user_msg, bot_msg]]
    formatted_history = []
    for user_msg, bot_msg in history:
        formatted_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            formatted_history.append({"role": "assistant", "content": bot_msg})

    # Get the bot's response
    response = bot(user_message, formatted_history)

    # Run the evaluation engine
    initial_state = {
        "question": user_message,
        "answer": response,
    }

    try:
        final_state = eval_app.invoke(initial_state)

        # Build the evaluation summary string
        eval_summary = "### Evaluation Results\n\n"

        # Metrics
        scores = {}
        if "clarity_results" in final_state and final_state["clarity_results"]:
            for metric, value in final_state["clarity_results"].items():
                if isinstance(value, (int, float, str)) and str(value).isdigit():
                    scores[metric] = float(value)

        if "pedagogy_results" in final_state and final_state["pedagogy_results"]:
            for metric, value in final_state["pedagogy_results"].items():
                if isinstance(value, (int, float, str)) and str(value).isdigit():
                    scores[metric] = float(value)

        if "statistics_results" in final_state and final_state["statistics_results"]:
            for metric, value in final_state["statistics_results"].items():
                if isinstance(value, (int, float)):
                    scores[metric] = round(value, 2)

        if "tone_results" in final_state and final_state["tone_results"]:
            for metric, value in final_state["tone_results"].items():
                if isinstance(value, (int, float, str)) and str(value).isdigit():
                    scores[metric] = float(value)

        eval_summary += "**Scores:**\n"
        for metric, score in scores.items():
            eval_summary += f"- {metric}: {score}\n"

        eval_summary += "\n**Fact Check:**\n"
        if "fact_checker_results" in final_state and final_state["fact_checker_results"]:
            for result in final_state["fact_checker_results"]:
                status = result.get('status', 'Unknown')
                emoji = "✅" if status == "Likely True" else "❌" if status == "Likely False" else "⚠️"
                eval_summary += f"{emoji} {result.get('fact')} ({status})\n"
        else:
            eval_summary += "No factual claims detected.\n"

    except Exception as e:
        eval_summary = f"### Evaluation Failed\nError: {str(e)}"

    return response, eval_summary

with gr.Blocks(title="Chatbot & Evaluation Dashboard") as demo:
    gr.Markdown("# 🤖 Advisor Chatbot & Real-time Evaluation")
    gr.Markdown("Chat with the Advisor persona. Each response will be evaluated in the background on Clarity, Pedagogy, Tone, Statistics, and Factuality.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation")
            msg = gr.Textbox(label="Type your message here...", placeholder="E.g., What is the 50/30/20 rule?")
            clear = gr.Button("Clear")

        with gr.Column(scale=1):
            eval_output = gr.Markdown(label="Evaluation Metrics", value="### Evaluation Results\nSubmit a message to see evaluation metrics for the response.")

    def respond(message, chat_history):
        bot_response, eval_results = chat_and_evaluate(message, chat_history)
        chat_history.append((message, bot_response))
        return "", chat_history, eval_results

    msg.submit(respond, [msg, chatbot], [msg, chatbot, eval_output])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
