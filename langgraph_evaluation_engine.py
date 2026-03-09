# langgraph_evaluation_engine.py

from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from evaluation_tools import (
    LLMClarityEvaluator,
    LLMPedagogyEvaluator,
    EvidentlyResponseEvaluatorTool,
    FactCheckTool,
    LLMToneEvaluator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LangGraphEvaluation")

class GraphState(TypedDict):
    """Represents the state of our evaluation graph."""
    question: str
    answer: str
    # FIX: Removed api_key from the state
    facts: List[str]
    clarity_results: Dict[str, Any]
    pedagogy_results: Dict[str, Any]
    statistics_results: Dict[str, Any]
    fact_checker_results: List[Dict[str, Any]]
    tone_results: Dict[str, Any]
    error: str

def extract_facts(state: GraphState) -> Dict[str, Any]:
    """Extracts facts from the answer to be used by the fact-checker."""
    logger.info("--- Extracting facts ---")
    answer = state["answer"]
    sentences = answer.split('.')
    facts = []
    factual_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will', 'contains', 'includes']
    for sentence in sentences:
        sentence = sentence.strip()
        if (sentence and len(sentence) > 15 and any(indicator in sentence.lower() for indicator in factual_indicators)):
            facts.append(sentence)
    return {"facts": facts[:4]}

# FIX: Node functions no longer handle or pass an api_key
def run_clarity_evaluator(state: GraphState) -> Dict[str, Any]:
    """Runs the clarity and conciseness evaluation tool."""
    logger.info("--- Running clarity evaluator ---")
    question = state["question"]
    answer = state["answer"]
    clarity_tool = LLMClarityEvaluator()
    results = clarity_tool.forward(question=question, answer=answer)
    return {"clarity_results": results}

def run_pedagogy_evaluator(state: GraphState) -> Dict[str, Any]:
    """Runs the pedagogical evaluation tool."""
    logger.info("--- Running pedagogy evaluator ---")
    question = state["question"]
    answer = state["answer"]
    pedagogy_tool = LLMPedagogyEvaluator()
    results = pedagogy_tool.forward(question=question, answer=answer)
    return {"pedagogy_results": results}

def run_statistics_evaluator(state: GraphState) -> Dict[str, Any]:
    """Runs the statistical evaluation tool."""
    logger.info("--- Running statistics evaluator ---")
    question = state["question"]
    answer = state["answer"]
    stats_tool = EvidentlyResponseEvaluatorTool()
    results = stats_tool.forward(prompt=question, response=answer)
    return {"statistics_results": results}

def run_fact_checker(state: GraphState) -> Dict[str, Any]:
    """Runs the fact-checking tool."""
    logger.info("--- Running fact-checker ---")
    facts_json = json.dumps(state["facts"])
    fact_checker_tool = FactCheckTool()
    results = fact_checker_tool.forward(facts_json)
    return {"fact_checker_results": json.loads(results)}

# The aggregate_and_visualize_results function remains the same as your version

def run_tone_evaluator(state: GraphState) -> Dict[str, Any]:
    """Runs the tone evaluation tool."""
    logger.info("--- Running tone evaluator ---")
    question = state["question"]
    answer = state["answer"]
    tone_tool = LLMToneEvaluator()
    results = tone_tool.forward(question=question, answer=answer)
    return {"tone_results": results}

def aggregate_and_visualize_results(state: GraphState) -> Dict[str, Any]:
    logger.info("--- Aggregating all evaluation results and generating visualizations ---")

    # Collect all numerical scores
    scores = {}

    # Extract from clarity results
    if "clarity_results" in state and state["clarity_results"]:
        for metric, value in state["clarity_results"].items():
            if isinstance(value, (int, float, str)) and str(value).isdigit():
                scores[metric] = float(value)

    # Extract from pedagogy results
    if "pedagogy_results" in state and state["pedagogy_results"]:
        for metric, value in state["pedagogy_results"].items():
            if isinstance(value, (int, float, str)) and str(value).isdigit():
                scores[metric] = float(value)

    # Extract from statistics results
    if "statistics_results" in state and state["statistics_results"]:
        for metric, value in state["statistics_results"].items():
            if isinstance(value, (int, float)):
                scores[metric] = value

    # Extract from tone results
    if "tone_results" in state and state["tone_results"]:
        for metric, value in state["tone_results"].items():
            if isinstance(value, (int, float, str)) and str(value).isdigit():
                scores[metric] = float(value)

    print("\n\n📊 EVALUATION SUMMARY:")
    for metric, score in scores.items():
        print(f"  - {metric}: {score}")

    print("\n🔍 FACT CHECK RESULTS:")
    if "fact_checker_results" in state and state["fact_checker_results"]:
        for result in state["fact_checker_results"]:
            status = result.get('status', 'Unknown')
            emoji = "✅" if status == "Likely True" else "❌" if status == "Likely False" else "⚠️"
            print(f"  {emoji} Fact: {result.get('fact')}")
            print(f"      Status: {status}")

    return {}

# --- Graph Assembly ---
workflow = StateGraph(GraphState)

workflow.add_node("extract_facts", extract_facts)
workflow.add_node("run_clarity_evaluator", run_clarity_evaluator)
workflow.add_node("run_pedagogy_evaluator", run_pedagogy_evaluator)
workflow.add_node("run_statistics_evaluator", run_statistics_evaluator)
workflow.add_node("run_fact_checker", run_fact_checker)
workflow.add_node("run_tone_evaluator", run_tone_evaluator)
workflow.add_node("aggregate_and_visualize", aggregate_and_visualize_results)

workflow.set_entry_point("extract_facts")

workflow.add_edge("extract_facts", "run_clarity_evaluator")
workflow.add_edge("extract_facts", "run_pedagogy_evaluator")
workflow.add_edge("extract_facts", "run_statistics_evaluator")
workflow.add_edge("extract_facts", "run_fact_checker")
workflow.add_edge("extract_facts", "run_tone_evaluator")

workflow.add_edge("run_clarity_evaluator", "aggregate_and_visualize")
workflow.add_edge("run_pedagogy_evaluator", "aggregate_and_visualize")
workflow.add_edge("run_statistics_evaluator", "aggregate_and_visualize")
workflow.add_edge("run_fact_checker", "aggregate_and_visualize")
workflow.add_edge("run_tone_evaluator", "aggregate_and_visualize")

workflow.add_edge("aggregate_and_visualize", END)

app = workflow.compile()