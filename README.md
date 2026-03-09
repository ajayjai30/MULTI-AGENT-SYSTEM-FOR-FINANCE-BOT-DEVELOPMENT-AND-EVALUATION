# AIML Project - Chatbot with Evaluation Workflow

## Overview

This project implements a chatbot system with a comprehensive evaluation workflow. The system generates responses and evaluates them using multiple evaluation tools.

## File Descriptions

### bot_agent.py

Creates a chatbot function that can be initialized in two ways:

- Using a bot_config dictionary with prompt_builder to create detailed prompts
- Using a custom_prompt string directly as system prompt
  Uses get_llm() from llm_wrapper for language model interaction

### langgraph_evaluation_engine.py

Implements a state graph workflow using LangGraph for evaluating responses. Contains:

- GraphState TypedDict defining evaluation state structure
- Node functions: extract_facts, run_clarity_evaluator, run_pedagogy_evaluator, run_statistics_evaluator, run_fact_checker
- Evaluation tools: LLMClarityEvaluator, LLMPedagogyEvaluator, EvidentlyResponseEvaluatorTool, FactCheckTool
- Workflow assembly with edges connecting evaluation nodes

### evaluation_tools.py

Contains evaluation tool implementations:

- FactCheckTool: Verifies factual claims using web search via DDGS
- EvidentlyResponseEvaluatorTool: Evaluates responses on length, sentence count, sentiment, relevance, hallucination
- LLMPedagogyEvaluator: Evaluates pedagogical quality with rubrics for critical thinking and term clarity
- LLMClarityEvaluator: Evaluates clarity, conciseness, completeness with specific rubrics
- LLMEvaluatorBase: Base class for LLM-based evaluators

### prompt_builder.py

Contains build_prompt function used by bot_agent.py to create detailed prompts from bot configuration

### question_generator.py

Generates questions for the evaluation agent to evaluate the chatbot

### test_bot.py

Testing file for the bot creation part of the application, may be even considered as the organized file for whole chat bot creation part

### test_evaluation.py

Testing file for the evaluation part of the application, may be even considered as the organized file for whole chat bot evaluation part, but instead of the example given it should be replaced with actual chat of the bot when asked with questions from question generator

### llm_wrapper.py

Contains get_llm() function used by bot_agent.py for language model interaction

### config.py

Configuration file for the llm's for the application

## Key Components

### Evaluation Workflow

1. extract_facts: Extracts factual claims from responses
2. run_clarity_evaluator: Evaluates response clarity using LLMClarityEvaluator
3. run_pedagogy_evaluator: Evaluates pedagogical quality using LLMPedagogyEvaluator
4. run_statistics_evaluator: Evaluates statistical metrics using EvidentlyResponseEvaluatorTool(This should be improved as it takes more time than other parts of evaluation try to do it without sentence transformers it will drastically reduce the time)
5. run_fact_checker: Verifies facts using FactCheckTool
6. aggregate_and_visualize_results: Aggregates and visualizes all evaluation results

### Evaluation Tools

- Fact checking via web search (DDGS)
- LLM-based evaluation using mistralai/mistral-7b-instruct:free model
- Multiple evaluation dimensions: clarity, pedagogy, statistics, factuality

## Usage

The system creates a chatbot using create_bot_chain() from bot_agent.py, then evaluates responses through the LangGraph evaluation workflow defined in langgraph_evaluation_engine.py.

## Dependencies

All dependencies are listed in requirements.txt

## Upgrades Needed:

1. Improve the speed of EvidentlyResponseEvaluatorTool by optimizing or replacing sentence transformers.
2. Enhance the FactCheckTool to handle a wider range of factual claims and improve accuracy.
3. Add more evaluation dimensions or tools as needed for comprehensive assessment.
4. Trying of giving access to tools for the chatbot created on the game.
