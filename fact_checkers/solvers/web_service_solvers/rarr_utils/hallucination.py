"""Utils for generating fake evidence given a query."""
import os
import time
from typing import Dict

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def run_evidence_hallucination(
    query: str,
    model: str,
    prompt: str,
    num_retries: int = 5,
) -> Dict[str, str]:
    """Generates a fake piece of evidence via LLM given the question.

    Args:
        query: Query to guide the validity check.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        output: A potentially inaccurate piece of evidence.
    """
    gpt3_input = prompt.format(query=query).strip()
    for _ in range(num_retries):
        try:
            response = openai.completions.create(
                model=model,
                prompt=gpt3_input,
                temperature=0.0,
                max_tokens=256,
                stop=["\n", "\n\n"],
            )
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(2)

    hallucinated_evidence = response.choices[0].text.strip()
    output = {"text": hallucinated_evidence, "query": query}
    return output
