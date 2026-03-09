# evaluation_tools.py
import os
import json
import pandas as pd
from smolagents import Tool
from duckduckgo_search import DDGS
from evidently.llm.templates import MulticlassClassificationPromptTemplate
from evidently import Dataset, DataDefinition
from evidently.descriptors import LLMEval, TextLength, Sentiment, SentenceCount
from openai import OpenAI

from llm_wrapper import get_llm

# This tool's logic is from your notebook
class FactCheckTool(Tool):
    name = "web_fact_checker"
    description = "Use this tool to verify a list of factual claims using web search and return an evaluation matrix."
    inputs = { "input": { "type": "string", "description": "A JSON list of factual claims." } }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.llm = get_llm()

    def _search_web(self, query):
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='us-en', max_results=5))
        return results

    def _evaluate_fact(self, fact):
        search_results = self._search_web(fact)
        context_snippets = [res["body"] for res in search_results if "body" in res]
        if not context_snippets:
            return {"fact": fact, "status": "Unverifiable", "evidence": "No relevant results"}

        combined_context = " ".join(context_snippets)

        system_prompt = "You are a factual verification assistant. Your task is to verify a claim against the provided search results. Respond strictly in JSON format with exactly two keys: 'status' (which must be exactly one of: 'Likely True', 'Partially True', 'Likely False', or 'Unverifiable') and 'reasoning' (a brief explanation of your decision based on the search results)."

        user_prompt = f"Claim: {fact}\nSearch Results Context: {combined_context}\nVerify the claim and return the JSON response."

        try:
            llm_response = self.llm(system_prompt_content=system_prompt, user_content=user_prompt)
            clean_response = llm_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]

            evaluation = json.loads(clean_response)
            status = evaluation.get("status", "Unverifiable")
            if status not in ["Likely True", "Partially True", "Likely False", "Unverifiable"]:
                status = "Unverifiable"

            return {"fact": fact, "status": status, "evidence": context_snippets[:2]}
        except Exception as e:
            # Fallback
            combined_context_lower = combined_context.lower()
            fact_lower = fact.lower()
            if fact_lower in combined_context_lower:
                return {"fact": fact, "status": "Likely True", "evidence": context_snippets[:2]}
            elif any(keyword in combined_context_lower for keyword in fact_lower.split()[:3]):
                return {"fact": fact, "status": "Partially True", "evidence": context_snippets[:2]}
            else:
                return {"fact": fact, "status": "Likely False", "evidence": context_snippets[:2]}

    def forward(self, input):
        try:
            facts = json.loads(input)
            assert isinstance(facts, list)
        except:
            return "❌ Invalid input. Please provide a JSON list of factual claims."
        evaluation_matrix = [self._evaluate_fact(fact) for fact in facts]
        return json.dumps(evaluation_matrix, indent=2)

# This tool's logic is from your notebook
class EvidentlyResponseEvaluatorTool(Tool):
    name = "evidently_response_evaluator"
    description = "Evaluates a prompt and response using multiple metrics: length, sentence count, sentiment, relevance, and hallucination."
    inputs = {
        "prompt": { "type": "string", "description": "The prompt that was given to the model." },
        "response": { "type": "string", "description": "The response from the model." }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        self.descriptors = [
            TextLength("answer", alias="Length"),
            SentenceCount("answer", alias="Sentence Count"),
            Sentiment("answer", alias="Sentiment Score")
        ]
        self.data_definition = DataDefinition(text_columns=["question", "answer"])

    def forward(self, prompt: str, response: str):
        if not (isinstance(prompt, str) and isinstance(response, str)):
            raise ValueError("Both 'prompt' and 'response' must be provided as strings.")
        eval_df = pd.DataFrame([{"question": prompt, "answer": response}])
        eval_dataset = Dataset.from_pandas(eval_df, data_definition=self.data_definition)
        eval_dataset.add_descriptors(self.descriptors)
        result_df = eval_dataset.as_dataframe()
        return {desc.alias: result_df[desc.alias].iloc[0] for desc in self.descriptors}

# This base class contains the definitive fix for the 401 Authentication Error
# In evaluation_tools.py

class LLMEvaluatorBase(Tool):
    inputs = {
        "question": { "type": "string", "description": "Prompt given to model." },
        "answer": { "type": "string", "description": "Response from model." }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        # The client will now be handled by the main script
        self.templates = {}
        self.descriptors = []
        self._setup_templates()

    def _setup_templates(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, question: str, answer: str):
        if not (isinstance(question, str) and isinstance(answer, str)):
            raise ValueError("Inputs must include 'question' and 'answer'.")

        eval_df = pd.DataFrame([[question, answer]], columns=["question", "answer"])
        data_definition = DataDefinition(text_columns=["question", "answer"])
        eval_dataset = Dataset.from_pandas(eval_df, data_definition=data_definition)
        eval_dataset.add_descriptors(self.descriptors)
        result_df = eval_dataset.as_dataframe()
        return {desc.alias: result_df[desc.alias].iloc[0] for desc in self.descriptors}

# ... (The other tool classes in the file remain the same) ...
# This tool's logic and rubrics are from your notebook
class LLMPedagogyEvaluator(LLMEvaluatorBase):
    name = "llm_pedagogy_evaluator"
    description = "Evaluates an LLM response on pedagogical rubrics."

    def _setup_templates(self):
        self.critical_thinking_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are an expert in educational psychology. Your task is to evaluate if a response encourages active learning and critical thinking.")],
            criteria="Analyze if the response prompts the user to question, analyze, and apply information, going beyond passive information delivery.",
            category_criteria={
                "5": "The response actively stimulates critical thinking with thought-provoking, open-ended questions or problem-solving scenarios.",
                "4": "The response includes some questions or prompts for reflection, but they may be superficial.",
                "3": "The response is purely informational, providing information passively.",
                "2": "The response is overly simplistic or provides closed answers that inhibit further inquiry.",
                "1": "The response consists only of questions without providing foundational information."
            }
        )
        self.term_clarity_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are an expert in technical writing. Your task is to evaluate if the response defines key terms before using them.")],
            criteria="Analyze if the response clearly defines key terminology before or as it is used, avoiding jargon to ensure the content is accessible.",
            category_criteria={
                "5": "The response proactively identifies and clearly defines all key terms before they are used in a complex context.",
                "4": "The response defines most key terms but might miss a few or provide slightly unclear definitions.",
                "3": "The response uses specialized terminology extensively without providing any definitions.",
                "2": "The response attempts to define terms, but the definitions are confusing or inaccurate.",
                "1": "The response avoids using any specialized terms that require a definition."
            }
        )
        self.descriptors = [
            LLMEval(
                column_name="answer", template=self.critical_thinking_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Promotes Critical Thinking"
            ),
            LLMEval(
                column_name="answer", template=self.term_clarity_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Terms Clearly Defined"
            ),
        ]

# This tool's logic and rubrics are from your notebook
class LLMClarityEvaluator(LLMEvaluatorBase):
    name = "clarity_evaluator"
    description = "Evaluates an LLM response on its clarity, conciseness, and completeness."

    def _setup_templates(self):
        self.core_message_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are an expert in communication. Your task is to evaluate how easily the core message of a response can be understood.")],
            criteria="Analyze how easily and quickly a user can understand the core message. It should be concise, clear, and not buried under jargon or rambling. ",
            category_criteria={
                "5": "The core message is stated upfront and is exceptionally clear and concise. The user can grasp the main point in seconds.",
                "4": "The core message is clear, but the user needs to read most of the response to understand it.",
                "3": "The core message is present but is buried under irrelevant details, requiring significant effort to find.",
                "2": "The response is unfocused or convoluted, making it difficult to determine the core message.",
                "1": "The response fails to deliver a core message."
            }
        )
        self.sentence_length_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are an expert editor focused on clarity. Your task is to evaluate sentence structure in a response.")],
            criteria="Analyze the sentence structure. Is it unnecessarily long or convoluted? The complexity should be appropriate for the topic. ",
            category_criteria={
                "5": "Sentences are well-constructed, direct, and easy to parse, enhancing readability.",
                "4": "Generally clear, but some sentences are longer than necessary or contain filler phrases.",
                "3": "Sentences are grammatically correct but too long, making them difficult to follow.",
                "2": "Sentences use overly complex syntax or jargon that obscures the meaning.",
                "1": "Sentences are both excessively long and confusingly structured."
            }
        )
        self.completeness_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are a detail-oriented analyst. Your task is to assess how thoroughly a response addresses all parts of a user's question.")],
            criteria="Evaluate if the response addresses all explicit and implicit parts of the user's question. ",
            category_criteria={
                "5": "The response comprehensively addresses every single part of the user's question.",
                "4": "The response addresses the main parts of the question but overlooks a minor detail.",
                "3": "The response addresses some significant parts but ignores other significant components.",
                "2": "The response focuses on only one aspect of the query, ignoring other parts.",
                "1": "The response fails to address any specific parts of the user's question."
            }
        )
        self.descriptors = [
            LLMEval(
                column_name="answer", template=self.core_message_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Core Message Clarity"
            ),
            LLMEval(
                column_name="answer", template=self.sentence_length_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Sentence Conciseness"
            ),
            LLMEval(
                column_name="answer", template=self.completeness_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Completeness of Answer"
            ),
        ]
# Added as an additional evaluation dimension
class LLMToneEvaluator(LLMEvaluatorBase):
    name = "tone_evaluator"
    description = "Evaluates an LLM response on its tone and empathy."

    def _setup_templates(self):
        self.empathy_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are an expert in communication. Your task is to evaluate the level of empathy in a response.")],
            criteria="Analyze the empathy level of the response. Does it acknowledge the user's feelings or situation?",
            category_criteria={
                "5": "The response is highly empathetic, validating the user's situation and providing supportive language.",
                "4": "The response is generally empathetic and supportive.",
                "3": "The response is neutral, focusing primarily on facts without much emotional acknowledgment.",
                "2": "The response is somewhat cold or dismissive of the user's context.",
                "1": "The response is completely apathetic or inappropriate for the user's situation."
            }
        )
        self.professionalism_template = MulticlassClassificationPromptTemplate(
            pre_messages=[("system", "You are an expert editor focused on professional communication. Your task is to evaluate the professionalism of a response.")],
            criteria="Analyze the tone of the response for professionalism and appropriate language.",
            category_criteria={
                "5": "The response is exceptionally professional, courteous, and appropriate.",
                "4": "The response is professional and polite.",
                "3": "The response is casual but acceptable.",
                "2": "The response is overly informal or slightly unprofessional.",
                "1": "The response is unprofessional, rude, or uses inappropriate language."
            }
        )
        self.descriptors = [
            LLMEval(
                column_name="answer", template=self.empathy_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Empathy Level"
            ),
            LLMEval(
                column_name="answer", template=self.professionalism_template,
                model="mistralai/mistral-7b-instruct:free", provider="openai", alias="Professionalism"
            ),
        ]
