from bot_agent import create_bot_chain

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

bot = create_bot_chain(bot_config)
conversation_history = [] # Initialize an empty history

print("\n🤖 Hi, I am your Advisor bot! Ask me anything about budgeting or goal setting.\n(Type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Pass the current history to the bot
    response = bot(user_input, conversation_history)
    print("🤖", response.strip(), "\n")

    # Update the history with the latest turn
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})