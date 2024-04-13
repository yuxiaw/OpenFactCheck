"""Utils for running the agreement gate."""
import os
import time
from typing import Any, Dict, Tuple

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_api_response(api_response: str) -> Tuple[bool, str, str]:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Agreement gate response from GPT-3.
    Returns:
        is_open: Whether the agreement gate is open.
        reason: The reasoning for why the agreement gate is open or closed.
        decision: The decision of the status of the gate in string form.
    """
    api_response = api_response.strip().split("\n")
    if len(api_response) < 2:
        reason = "Failed to parse."
        decision = None
        is_open = False
    else:
        reason = api_response[0]
        decision = api_response[1].split("Therefore:")[-1].strip()
        is_open = "disagrees" not in api_response[1]
    return is_open, reason, decision


def run_agreement_gate(
    claim: str,
    query: str,
    evidence: str,
    model: str,
    prompt: str,
    context: str = None,
    num_retries: int = 5,
) -> Dict[str, Any]:
    """Checks if a provided evidence contradicts the claim given a query.

    Checks if the answer to a query using the claim contradicts the answer using the
    evidence. If so, we open the agreement gate, which means that we allow the editor
    to edit the claim. Otherwise the agreement gate is closed.

    Args:
        claim: Text to check the validity of.
        query: Query to guide the validity check.
        evidence: Evidence to judge the validity of the claim against.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        gate: A dictionary with the status of the gate and reasoning for decision.
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
                max_tokens=256,
                stop=["\n\n"],
                logit_bias={"50256": -100},  # Don't allow <|endoftext|> to be generated
            )
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(2)

    is_open, reason, decision = parse_api_response(response.choices[0].text)
    gate = {"is_open": is_open, "reason": reason, "decision": decision}
    return gate
