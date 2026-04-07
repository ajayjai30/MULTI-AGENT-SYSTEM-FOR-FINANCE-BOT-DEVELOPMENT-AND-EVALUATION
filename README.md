## MULTI-AGENT WORKFLOW - FOR CHATBOT EVALUATIONS


## Overview

This project implements a chatbot system with a comprehensive, multi-agent evaluation workflow. The system dynamically generates persona-driven chatbot responses and seamlessly evaluates them using several specialized agents.

## How it works

The core is structured around `bot_agent.py`, which generates a conversational AI based on a specified persona or configuration. The chatbot runs live and updates its context. The generated interactions are then tested through `langgraph_evaluation_engine.py`, which acts as the orchestrator for multiple evaluation nodes. It uses LangGraph's StateGraph to manage the workflow, passing the response and question into each evaluation step and aggregating the results at the end.

## File Descriptions

### `bot_agent.py`

Creates a chatbot function that can be initialized in two ways:
- Using a `bot_config` dictionary with `prompt_builder` to create detailed prompts.
- Using a `custom_prompt` string directly as a system prompt.

### `langgraph_evaluation_engine.py`

Implements a state graph workflow using LangGraph for evaluating responses. Contains:
- `GraphState`: A TypedDict defining evaluation state structure.
- Node functions: `extract_facts`, `run_clarity_evaluator`, `run_pedagogy_evaluator`, `run_statistics_evaluator`, `run_fact_checker`, and `run_tone_evaluator`.
- Evaluation tools: `LLMClarityEvaluator`, `LLMPedagogyEvaluator`, `EvidentlyResponseEvaluatorTool`, `FactCheckTool`, and `LLMToneEvaluator`.
- Workflow assembly with edges connecting evaluation nodes.

### `evaluation_tools.py`

Contains specialized evaluation tool implementations:
- **`FactCheckTool`**: Verifies factual claims using web search via DuckDuckGo (DDGS) combined with LLM analysis for enhanced accuracy.
- **`EvidentlyResponseEvaluatorTool`**: Evaluates responses primarily on length, sentence count, and sentiment using the `evidently` library. (Optimized by removing slow semantic similarity).
- **`LLMPedagogyEvaluator`**: Evaluates pedagogical quality using prompt rubrics for critical thinking and term clarity.
- **`LLMClarityEvaluator`**: Evaluates clarity, conciseness, and completeness.
- **`LLMToneEvaluator`**: Assesses response tone for empathy and professionalism.
- **`LLMEvaluatorBase`**: Base class for all LLM-based evaluators.

### `prompt_builder.py`
Contains the `build_prompt` function used by `bot_agent.py` to create detailed persona prompts from the bot configuration.

### `question_generator.py`
Generates sequential questions using an LLM to simulate a natural, multi-turn conversation tailored to the chatbot's domain for testing.

### `test_bot.py`
Testing script for interacting directly with the configured persona bot.

### `test_evaluation.py`
Testing script for evaluating the chatbot's live interactions using the generated questions and executing the LangGraph evaluation engine on each turn.

### `llm_wrapper.py`
Contains the `get_llm()` function to interact seamlessly with language models across the application.

### `config.py`
Configuration file containing the default and fallback LLMs for the application.

## Key Components

### Evaluation Workflow

1. `extract_facts`: Extracts factual claims from the model's responses.
2. `run_clarity_evaluator`: Assesses clarity using `LLMClarityEvaluator`.
3. `run_pedagogy_evaluator`: Assesses pedagogical quality using `LLMPedagogyEvaluator`.
4. `run_statistics_evaluator`: Assesses text length, sentence count, and sentiment.
5. `run_fact_checker`: Verifies facts using `FactCheckTool` with web search context.
6. `run_tone_evaluator`: Assesses response empathy and professionalism.
7. `aggregate_and_visualize_results`: Aggregates and prints all the extracted numerical scores and fact-checker statuses.

### Evaluation Tools

- Web Search Integration (DuckDuckGo).
- LLM-based verification using configurable models.
- Fast Statistical Metrics using the `evidently` library.

## Usage

Create a chatbot using `create_bot_chain()` from `bot_agent.py`, and evaluate the responses using the LangGraph engine defined in `langgraph_evaluation_engine.py` (Refer to `test_evaluation.py`).

## Dependencies

All required dependencies are listed in `requirements.txt`.
