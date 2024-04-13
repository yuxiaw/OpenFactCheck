"""Utils for running the editor."""
import os
import time
from typing import Dict, Union

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_api_response(api_response: str) -> str:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns a reason for the edit and the edit in two consecutive lines.
    Only extract out the edit from the second line.

    Args:
        api_response: Editor response from GPT-3.
    Returns:
        edited_claim: The edited claim.
    """
    api_response = api_response.strip().split("\n")
    if len(api_response) < 2:
        print("Editor error.")
        return None
    edited_claim = api_response[1].split("My fix:")[-1].strip()
    return edited_claim


def run_rarr_editor(
    claim: str,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    context: str = None,
    num_retries: int = 5,
) -> Dict[str, str]:
    """Runs a GPT-3 editor on the claim given a query and evidence to support the edit.

    Args:
        claim: Text to edit.
        query: Query to guide the editing.
        evidence: Evidence to base the edit on.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        edited_claim: The edited claim.
    """
    if context:
        gpt3_input = prompt.format(
            context=context, claim=claim, query=query, evidence=evidence
        ).strip()
    else:
        gpt3_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()

    for _ in range(num_retries):
        try:
            response = openai.completions.create(
                model=model,
                prompt=gpt3_input,
                temperature=0.0,
                max_tokens=512,
                stop=["\n\n"],
            )
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(2)

    edited_claim = parse_api_response(response.choices[0].text)
    # If there was an error in GPT-3 generation, return the claim.
    if not edited_claim:
        edited_claim = claim
    output = {"text": edited_claim}
    return output
