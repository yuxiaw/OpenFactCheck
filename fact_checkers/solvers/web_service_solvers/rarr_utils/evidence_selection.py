import itertools
from typing import Any, Dict, List

import torch
from sentence_transformers import CrossEncoder

PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


def compute_score_matrix(
    questions: List[str], evidences: List[str]
) -> List[List[float]]:
    """Scores the relevance of all evidence against all questions using a CrossEncoder.

    Args:
        questions: A list of unique questions.
        evidences: A list of unique evidences.
    Returns:
        score_matrix: A 2D list list of question X evidence relevance scores.
    """
    score_matrix = []
    for q in questions:
        evidence_scores = PASSAGE_RANKER.predict([(q, e) for e in evidences]).tolist()
        score_matrix.append(evidence_scores)
    return score_matrix


def question_coverage_objective_fn(
    score_matrix: List[List[float]], evidence_indices: List[int]
) -> float:
    """Given (query, evidence) scores and a subset of evidence, return the coverage.

    Given all pairwise query and evidence scores, and a subset of the evidence
    specified by indices, return a value indicating how well this subset of evidence
    covers (i.e., helps answer) all questions.

    Args:
        score_matrix: A 2D list list of question X evidence relevance scores.
        evidence_indicies: A subset of the evidence to to get the coverage score of.
    Returns:
        total: The coverage we would get by using the subset of evidence in
            `evidence_indices` over all questions.
    """
    # Compute sum_{question q} max_{selected evidence e} score(q, e).
    # This encourages all questions to be explained by at least one evidence.
    total = 0.0
    for scores_for_question in score_matrix:
        total += max(scores_for_question[j] for j in evidence_indices)
    return total


def select_evidences(
    example: Dict[str, Any], max_selected: int = 5, prefer_fewer: bool = False
) -> List[Dict[str, Any]]:
    """Selects the set of evidence that maximizes information converage over the claim.

    Args:
        example: The result of running the editing pipeline on one claim.
        max_selected: Maximum number of evidences to select.
        prefer_fewer: If True and the maximum objective value can be achieved by
            fewer evidences than `max_selected`, prefer selecting fewer evidences.
    Returns:
        selected_evidences: Selected evidences that serve as the attribution report.
    """
    questions = sorted(set(example["questions"]))
    evidences = sorted(set(e["text"] for e in example["revisions"][0]["evidences"]))
    num_evidences = len(evidences)
    if not num_evidences:
        return []

    score_matrix = compute_score_matrix(questions, evidences)

    best_combo = tuple()
    best_objective_value = float("-inf")
    max_selected = min(max_selected, num_evidences)
    min_selected = 1 if prefer_fewer else max_selected
    for num_selected in range(min_selected, max_selected + 1):
        for combo in itertools.combinations(range(num_evidences), num_selected):
            objective_value = question_coverage_objective_fn(score_matrix, combo)
            if objective_value > best_objective_value:
                best_combo = combo
                best_objective_value = objective_value

    selected_evidences = [{"text": evidences[idx]} for idx in best_combo]
    return selected_evidences
