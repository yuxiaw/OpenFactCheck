import argparse
import json
import os
import re
import pytz
import string
import torch
import datetime
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm, trange
from argparse import Namespace

# from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    f1_score,
    recall_score,
)
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from factool_benchmark import evaluate_free_text_by_factool, calcPrice, sumAllObj
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eval_classification(y_true, y_pred, average="macro"):
    precision, recall, F1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average
    )
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics


def eval_binary_classification(y_true, y_pred, pos_label="yes"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    F1 = f1_score(y_true, y_pred, pos_label=pos_label)

    metrics = {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "F1": round(F1, 3),
    }
    return metrics


def get_dataset_model_response(df: pd.DataFrame, dataset, model = "testmodel_response"):
    content = df

    # Better implementation:
    # responses = df.query("source == 'snowballing'")[['topic',model]]

    responses = []
    if dataset == "snowballing":
        for k, row in content.iterrows():
            if row["source"] == dataset:
                responses.append({"topic": row["topic"], "response": row[model]})

    # Better implementation:
    # responses = df.query(" | ".join( [ f"source == '{x}'" for x in selfaware_subset]))[['label_unanswerable',model]]

    elif dataset == "selfaware":
        selfaware_subset = [
            "selfaware-hotpot_train",
            "selfaware-squadqa_train",
            "selfaware-triviaqa_train",
            "selfaware-squadqa_dev",
            "selfaware-hotpot_dev",
            "selfaware-triviaqa_dev",
            "selfaware-SelfAware",
        ]
        for k, row in content.iterrows():
            if row["source"] in selfaware_subset:
                responses.append(
                    {
                        "label_unanswerable": row["ability_to_test"].lstrip(
                            "answerable: "
                        )
                                              == "False",
                        "response": row[model],
                    }
                )

    elif dataset == "freshqa":
        for k, row in content.iterrows():
            if row["source"] == dataset:
                responses.append(
                    {
                        "question": row["question"],
                        "ref_answer": row["reference_answer"],
                        "response": row[model],
                        "model": model,
                    }
                )
        print(model, len(responses))

    elif dataset in ["factool-qa", "felm-wk", "factcheckgpt"]:
        for k, row in content.iterrows():
            if row["source"] == dataset:
                responses.append(
                    {
                        "source": row["source"],
                        "prompt": row["prompt"],
                        "response": row[model],
                    }
                )

    print("Items: ", len(responses))
    return responses


# ------------------------------------------------------------------------------
# Evaluate LLM answers on Snowballing dataset
# ------------------------------------------------------------------------------
def contains_any_of_specified_words_regex(input_string, ll):
    pattern = r"{}".format("|".join(ll))
    return bool(re.search(pattern, input_string, re.IGNORECASE))


def get_yes_or_no(response, strict=False):
    low_response = response.lower()
    if strict:
        if low_response.startswith("yes"):
            return "yes"
        elif low_response.startswith("no"):
            return "no"
        return ""
    else:
        if contains_any_of_specified_words_regex(response, ["n't", "no"]):
            return "no"
        else:
            return "yes"


def snowballing_evaluation(
        llm_responses, intermediate_results_dir, response_column_name
):
    results = (
        {}
    )  # evaluation results over three topics and the whole dataset, key is topics

    topic_answers = {
        "Primality Testing": "yes",
        "US Senator Search": "yes",
        "Graph Connectivity-Flight Search": "no",
    }

    topic_responses = {}
    for key in topic_answers:
        topic_responses[key] = []

    for item in llm_responses:
        topic_responses[item["topic"]].append(get_yes_or_no(item["response"]))

    gold_labels, predictions = [], []
    for key in topic_answers:
        results[key] = eval_binary_classification(
            [topic_answers[key]] * len(topic_responses[key]), topic_responses[key]
        )
        gold_labels += [topic_answers[key]] * len(topic_responses[key])
        predictions += topic_responses[key]
    results["All"] = eval_binary_classification(
        gold_labels, predictions, pos_label="yes"
    )

    # save labels and preds
    df = pd.DataFrame({"gold_labels": gold_labels, "predictions": predictions})
    if not os.path.exists(intermediate_results_dir):
        os.mkdir(intermediate_results_dir)
    df.to_json(f"{intermediate_results_dir}/snowballing_{response_column_name}.json")

    return results, gold_labels, predictions


# ------------------------------------------------------------------------------
# Evaluate LLM answers on Selfaware dataset
# ------------------------------------------------------------------------------
def remove_punctuation(input_string):
    input_string = input_string.strip().lower()
    if input_string and input_string[-1] in string.punctuation:
        return input_string[:-1]
    return input_string


uncertain_list = [
    "The answer is unknown.",
    "The answer is uncertain.",
    "The answer is unclear.",
    "There is no scientific evidence.",
    "There is no definitive answer.",
    "There is no right answer.",
    "There is much debate.",
    "There is no known case.",
    "There is no concrete answer to this question.",
    "There is no public information available.",
    "It is impossible to know.",
    "It is impossible to answer.",
    "It is difficult to predict.",
    "It is not known.",
    "We do not know.",
    "I'm not sure.",
]
uncertain_list = [remove_punctuation(_) for _ in uncertain_list]


def cut_sentences(content):
    sentences = re.split(r"(\.|\!|\?|。|！|？|\.{6})", content)
    return sentences


def cut_sub_string(input_string, window_size=5, punctuation=".,?!"):
    input_string = input_string.strip().lower()
    if len(input_string) < 2:
        return [""]
    if input_string[-1] in punctuation:
        input_string = input_string[:-1]
    string_list = input_string.split()
    length = len(string_list)
    if length <= window_size:
        return [input_string]
    else:
        res = []
        for i in range(length - window_size + 1):
            sub_string = " ".join(string_list[i: i + window_size])
            if sub_string != "" or sub_string != " ":
                res.append(sub_string)
        return res


def group_similarity(model, tokenizer, sentences1, sentences2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded1 = tokenizer(
        sentences1, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    encoded2 = tokenizer(
        sentences2, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    embeddings1 = model(**encoded1).pooler_output
    embeddings2 = model(**encoded2).pooler_output

    # Calculate the cosine similarity
    similarities = cosine_similarity(
        embeddings1.detach().cpu().numpy(), embeddings2.detach().cpu().numpy()
    )
    return similarities


def get_unanswerable(response, model, tokenizer, threshold=0.75):
    pred_unanswerable = False
    response = response.lower()
    for uncertain in uncertain_list:
        if uncertain in response:
            pred_unanswerable = True

    if pred_unanswerable == False:
        sub_sen_list = cut_sentences(response)
        sub_str_list = []
        for sub_sen in sub_sen_list:
            if len(sub_sen) >= 2:
                sub_str_list.extend(cut_sub_string(sub_sen))
        if len(sub_str_list) != 0:
            # print(sub_str_list)
            similarities = group_similarity(
                model, tokenizer, sub_str_list, uncertain_list
            )

        else:
            similarities = [0]
        max_uncertainty = np.max(similarities)
        # print(max_uncertainty)
        if max_uncertainty > threshold:
            pred_unanswerable = True

    return pred_unanswerable


def selfaware_evaluation(llm_responses, intermediate_results_dir, response_column_name):
    model_name = "princeton-nlp/sup-simcse-roberta-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    preds, labels = [], []
    for item in llm_responses:
        # gold label: whether the question is answerable or not.
        labels.append(item["label_unanswerable"])
        # identify whether the model response answered the question or not.
        preds.append(get_unanswerable(item["response"], model, tokenizer))

    # map: answerable -> False, unanswerable -> True
    result1 = eval_binary_classification(labels, preds, pos_label=True)
    result2 = eval_binary_classification(labels, preds, pos_label=False)
    results = {
        "unanswerable_as_pos": result1,
        "answerable_as_pos": result2,
    }

    # save labels and preds
    df = pd.DataFrame({"gold_labels": labels, "predictions": preds})
    if not os.path.exists(intermediate_results_dir):
        os.mkdir(intermediate_results_dir)
    df.to_json(f"{intermediate_results_dir}/selfaware_{response_column_name}.json")

    return results, labels, preds


# ------------------------------------------------------------------------------
# Evaluate LLM answers on FreshQA dataset
# ------------------------------------------------------------------------------
openai_api_key = os.environ.get("OPENAI_API_KEY", None)

openai_client = OpenAI(
    api_key=openai_api_key,
)

# model_name = "gpt-4-turbo-preview"
model_name = "gpt-3.5-turbo"
current_date = datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime(
    "%B %d, %Y"
)
prefix = (
    "Please evaluate the response to a question under strict evaluation, where"
    " no hallucinations, outdated information, or ill-formed answers are"
    " allowed. Please credit the response only if it provides a confident and"
    " definitive answer, or the correct answer can be obviously inferred from"
    " the response. The primary or final answer when standing alone must be"
    " accurate. Any additional information that is provided must not contradict"
    " the primary answer or reshape one's perception of it. For false-premise"
    " questions, the response must point out the presence of a false premise to"
    " receive credit. For answers that involve names of entities (e.g.,"
    " people), complete names or commonly recognized names are expected."
    " Regarding numerical answers, approximate numbers are generally not"
    " accepted unless explicitly included in the ground-truth answers. A"
    " response that contains any hallucination, no matter how minor, will not"
    " receive credit. Furthermore, when the response indicates that the"
    " information might be outdated, we accept it only if it is evident that"
    " the knowledge has not changed (e.g., through common sense or well-known"
    " facts)."
)
demo_examples = [
    {
        "question": "How old is the world's oldest verified living person?",
        "correct_answers": [
            "116 years old",
            "116",
        ],
        "response": (
            f"As of today {current_date}, the most up-to-date and relevant"
            " information regarding this query is as follows. The oldest"
            " verified living person is Maria Branyas Morera, who was born"
            " on March 4, 1907, making her 116 years old."
        ),
        "comment": (
            "This is a valid question. The primary answer in the response"
            " (116 years old) is accurate, and all the provided information"
            " is accurate and up-to-date. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": "When did the UK adopt the Euro?",
        "correct_answers": ["The United Kingdom has never adopted the Euro."],
        "response": (
            "The UK has never adopted the Euro as its official currency. The"
            " country has retained the British pound sterling (GBP) as its"
            " currency."
        ),
        "comment": (
            "The question contains a false premise. The response debunks the"
            " question's false premise (i.e., it points out that the UK has"
            " never adopted the Euro), and all the provided information is"
            " accurate and up-to-date. Thus, the response is credited."
        ),
        "evaluation": "correct",
    },
    {
        "question": (
            "How much longer will Brittney Griner spend in the Russian prison?"
        ),
        "correct_answers": [
            "She was released in December 2022 as part of a prisoner swap."
        ],
        "response": (
            "I'm sorry, but I have no information to suggest that Brittney"
            " Griner is currently in a Russian prison. As far as I know, she"
            " is a professional basketball player who currently plays for the"
            " Phoenix Mercury in the Women's National Basketball Association"
            " (WNBA). If you have any more information or context, please let"
            " me know and I'll do my best to help."
        ),
        "comment": (
            "The question contains a false premise. The response fails to"
            " debunk the question's false premise by not explicitly stating"
            " that Brittney Griner was already released. Additionally, the"
            " response lacks a confident and definitive answer. Thus, the"
            " response is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": (
            "Which language has the third largest population by number of"
            " native speakers?"
        ),
        "correct_answers": ["English"],
        "response": "1. Mandarin 2. Spanish 3. English",
        "comment": (
            "This is a valid question. While the correct answer (English) can"
            " be inferred from the response, the response lacks an explicit"
            " statement, which is not desirable."
            " Thus, the response is not credited."
        ),
        "evaluation": "incorrect",
    },
    {
        "question": (
            "Has Virginia Woolf's novel about the Ramsay family entered the"
            " public domain in the United States?"
        ),
        "correct_answers": ["Yes"],
        "response": (
            "1927's To the Lighthouse is in the public domain in the United"
            " States. The final answer is To the Lighthouse."
        ),
        "comment": (
            "This is a valid question. While the primary information in the"
            " response (1927's To the Lighthouse is in the public domain in"
            " the United States) is accurate, the final answer when standing"
            ' alone is inaccurate (it should be "Yes" instead of "To the'
            ' Lighthouse"). Thus, the response is not credited.'
        ),
        "evaluation": "incorrect",
    },
]

demo_questions = [ex["question"] for ex in demo_examples]
demo_evaluation_template = (
    "\ncorrect answer(s): {correct_answers}"
    "\nresponse: {response}"
    "\ncomment: {comment}"
    "\nevaluation: {evaluation}"
)
evaluation_template = (
    "\ncorrect answer(s): {correct_answers}" "\nresponse: {response}" "\ncomment: "
)

demo_evaluations = []
for ex in demo_examples:
    demo_evaluation = demo_evaluation_template.format(
        question=ex["question"],
        correct_answers=" | ".join(ex["correct_answers"]),
        response=ex["response"],
        comment=ex["comment"],
        evaluation=ex["evaluation"],
    )
    demo_evaluations.append(demo_evaluation)


def call_llm_api(prompt, model, temperature, max_tokens, chat_completions=True):
    # See https://platform.openai.com/docs/guides/gpt for details
    if chat_completions:
        response = openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Respond as concisely as"
                        f" possible. Knowledge cutoff: {current_date}."
                    ),
                },
                {"role": "user", "content": "What's today's date?"},
                {
                    "role": "assistant",
                    "content": f"Today is {current_date} in Pacific Standard Time.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = openai_client.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
        )
        return response.choices[0].text


def call_fresheval(model, prefix, question, response, correct_answers, evaluation):
    temperature = 0.0
    max_tokens = 256
    chat_completions = True

    if model.startswith("gpt-4"):
        num_organic_results = 15
        num_related_questions = 3
        num_questions_and_answers = 3
        num_retrieved_evidences = 15
    else:
        num_organic_results = 15
        num_related_questions = 2
        num_questions_and_answers = 2
        num_retrieved_evidences = 5

    # Generate prompts for demo examples
    demo_prompts = []
    for q, e in zip(demo_questions, demo_evaluations):
        demo_prompts.append(f"\n\n\nquestion: {q}{e}")

    fresheval_demo = "".join(demo_prompts).strip()
    fresheval_question = f"\n\n\nquestion: {question}{evaluation}"

    fresh_eval = prefix + "\n\n\n" + fresheval_demo + fresheval_question

    answer = call_llm_api(fresh_eval, model, temperature, max_tokens, chat_completions)

    return answer


def save_list_to_file(filename, data):
    df = pd.DataFrame({"raw_eval": data[0], "eval_rating": data[1]})
    df.to_csv(filename, index=False)


def add_line_to_file(filename, new_line):
    df = pd.read_csv(filename)
    new_df = pd.DataFrame({"raw_eval": new_line[0], "eval_rating": new_line[1]})
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(filename, index=False)


def extract_ratings(response):
    # if the eval answer contains either of these three words, considered as 0
    # including incorrect, not correct, not credited
    pattern = re.compile(
        r"\b(?:incorrect|not\s+correct|not\s+credited)\b", re.IGNORECASE
    )
    if pattern.search(response):
        return 0
    else:
        return 1


def get_fresheval_res(question, correct_answers, response, logfile):
    evaluation = evaluation_template.format(
        correct_answers=correct_answers,
        response=response,
    )

    fresheval = call_fresheval(
        model_name,
        prefix,
        question,
        response,
        correct_answers,
        evaluation,
    )

    eval = extract_ratings(fresheval)

    # save LLM evaluation results and extracted rating
    filename = logfile
    try:
        # Adding a new line
        add_line_to_file(filename, [fresheval, {"rating": eval}])
    except:
        # Saving initial list to file
        save_list_to_file(filename, [fresheval, {"rating": eval}])
    return eval


def fresh_evaluation(llm_responses, intermediate_results_dir, response_column_name):
    logfile = f"freshqa_{response_column_name}.csv"
    llm_responses = pd.DataFrame(llm_responses)
    preds = []
    for idx, row in llm_responses.iterrows():
        eval = get_fresheval_res(
            row["question"],
            row["ref_answer"],
            row["response"],
            os.path.join(intermediate_results_dir, logfile),
        )
        preds.append(eval)
    return sum(preds) / len(preds)


# ------------------------------------------------------------------------------
# Evaluate LLM answers on Free-style answer datasets by Factool
# ------------------------------------------------------------------------------
def collect_data(projectdir, response_column_name):
    model_name = response_column_name[:-9]
    print(model_name)
    data = []
    dirnames = [name for name in os.listdir(projectdir) if model_name in name]
    for dirname in dirnames:
        dirpath = os.path.join(projectdir, dirname)
        if os.path.isdir(dirpath):
            if os.path.exists(os.path.join(dirpath, "eval_result.json")):
                with open(os.path.join(dirpath, "eval_result.json"), "r") as f:
                    data.append(json.load(f))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question_df_dir",
        default="./question_set/evaluate_llm_factuality_dataset.jsonl",
        help="path of original questions with its properties jsonl file",
    )
    parser.add_argument(
        "--input_path",
        help="Path to response files",
        default="./model_response/GPT4_responses.csv",
    )
    parser.add_argument(
        "--eval_result_path",
        help="Path for saving intermediate results",
        default="./intermediate_results",
    )
    parser.add_argument(
        "--auto_checker_path",
        help="Path of automatic fact-checkers used to evaluate free-style model responses.",
        default="fact_checkers/solvers",
    )
    parser.add_argument(
        "--auto_checker_save_foldername",
        help="Path for saving automatic fact-checker's evaluation results",
        default="./factool_evaluation",
    )
    parser.add_argument(
        "--auto_checker",
        help="Name of automatic fact checker used for free-style response verification.",
        default="factool_solvers",
    )
    parser.add_argument(
        "--auto_checker_config",
        help="The configuration file of the automatic fact-checkers.",
        default="factool_config.yaml",
    )
    parser.add_argument(
        "--gpt_version",
        help="GPT version",
        default="gpt-4-0613",
    )
    parser.add_argument(
        "--datasets",
        help="The datasets used to evaluate LLM factuality.",
        default=[
            "snowballing",
            "selfaware",
            "freshqa",
            "factool-qa",
            "felm-wk",
            "factcheckgpt",
        ],
    )
    parser.add_argument(
        "--openai_apikey",
        help="OpenAI API key",
        default=os.environ.get("OPENAI_API_KEY", None),
    )
    args = parser.parse_args()

    # pass by args
    question_df_dir = args.question_df_dir
    test_responses_dir = args.input_path
    # response_column_name = args.response_key
    intermediate_results_dir = args.eval_result_path

    # Folder to save the results
    projectdir = os.path.join(
        intermediate_results_dir, args.auto_checker_save_foldername
    )
    os.makedirs(projectdir, exist_ok=True)
    examples_dir = os.path.join(os.path.dirname(os.getcwd()), args.auto_checker_path)
    configs_dir = os.path.join("/".join(examples_dir.split("/")[:-1]), "config")
    solver_args = Namespace(
        user_src=os.path.join(examples_dir, args.auto_checker),
        config=os.path.join(configs_dir, args.auto_checker_config),
        output=os.path.join(projectdir, "truth"),
        openai_apikey=args.openai_apikey,
    )

    # merge responses to test to df read from evaluate_llm_factuality_dataset.jsonl
    responses_to_test = pd.read_csv(test_responses_dir)
    # ask users to use modelname_response as column when uploading responses
    # get the column name: modelname_response, for example GPT4_response
    response_column_name = responses_to_test.columns.tolist()[-1]
    df = pd.read_json(question_df_dir, lines=True)
    try:
        assert len(responses_to_test) == len(df)
        # This is better implementation:
        # df = pd.concat([df, responses_to_test], axis=1)
        df["testmodel_response"] = list(responses_to_test[response_column_name])
    except AssertionError as e:
        print(f"AssertionError: {e}")
    except:
        print("There are some issues when merging responses to question sets.")

    # evaluate model responses over each dataset
    combined_result = []
    for dataset in args.datasets:
        responses = get_dataset_model_response(df, dataset)
        logger.info(f"Evaluating {dataset} with size of responses {len(responses)}")
        results = {}
        if dataset == "snowballing":
            results, gold_labels, predictions = snowballing_evaluation(
                responses, intermediate_results_dir, response_column_name
            )
            combined_result.append(results)
        elif dataset == "selfaware":
            results, labels, preds = selfaware_evaluation(
                responses[:50], intermediate_results_dir, response_column_name
            )
            combined_result.append(results)
        elif dataset == "freshqa":
            accuracy = fresh_evaluation(
                responses[:50], intermediate_results_dir, response_column_name
            )
            combined_result.append({"Accuracy": accuracy})
        else:
            evaluate_free_text_by_factool(responses[:50], response_column_name, args=solver_args, projectdir=projectdir)

            # read saved results and evaluate
            data = collect_data(projectdir, response_column_name)

            # False claims, USD and time cost
            (
                costs,
                time_costs,
                falseClaims,
                trueResponse,
                numResponse,
            ) = (
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            )
            for d in data:
                if d["llm"] == response_column_name:
                    if d["dataset"] == "factool-qa":
                        numResponse[0] += 1
                        costs[0] += calcPrice(sumAllObj(d["claims"]))
                        time_costs[0] += d["end"] - d["start"]
                        falseClaims[0] += d["claims"]["numFalseClaims"]
                        if (d["claims"]["numFalseClaims"] + d["claims"]["numMixedClaims"]) == 0:
                            trueResponse[0] += 1
                    elif d["dataset"] == "felm-wk":
                        numResponse[1] += 1
                        costs[1] += calcPrice(sumAllObj(d["claims"]))
                        time_costs[1] += d["end"] - d["start"]
                        falseClaims[1] += d["claims"]["numFalseClaims"]
                        if (d["claims"]["numFalseClaims"] + d["claims"]["numMixedClaims"]) == 0:
                            trueResponse[1] += 1
                    elif d["dataset"] == "factcheckgpt":
                        numResponse[2] += 1
                        costs[2] += calcPrice(sumAllObj(d["claims"]))
                        time_costs[2] += d["end"] - d["start"]
                        falseClaims[2] += d["claims"]["numFalseClaims"]
                        if (d["claims"]["numFalseClaims"] + d["claims"]["numMixedClaims"]) == 0:
                            trueResponse[2] += 1
                else:
                    print("Conflict between saved model name and model to test.")
            results = {
                "False Claims": falseClaims,
                "USD cost": costs,
                "Time cost (ms)": time_costs,
                "Percentage of true responses": [
                    round(trueResponse[i] / numResponse[i] if numResponse[i] != 0 else 0, 3) for i in range(3)
                ]
            }
            combined_result.append(results)
        logger.info(f"Evaluate results: {results}")

    # save all evaluation results
    with open(
            os.path.join(intermediate_results_dir, "combined_result.json"), "w"
    ) as json_file:
        json.dump(combined_result, json_file)


if __name__ == "__main__":
    main()
