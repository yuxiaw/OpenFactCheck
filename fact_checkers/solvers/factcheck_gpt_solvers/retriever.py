from core import *
from argparse import Namespace
from .utils.openai_api import gpt
import spacy
from sentence_transformers import CrossEncoder
import spacy
import numpy as np
from copy import deepcopy
import torch
import openai
import concurrent.futures
import backoff
import requests
import re
import itertools
from openai import RateLimitError
import bs4
from typing import List, Dict, Any
from .utils.prompt import QGEN_PROMPT, QGEN_PROMPT_FMT
from .utils.data_util import save_txt, save_json
import time


@register_solver("retriever", "claims", "claims_with_evidences")
class Retriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("model", "gpt-3.5-turbo")
        self.num_retries = self.global_config.get("num_retries", 3)
        self.tokenizer = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        self.question_duplicate_model = CrossEncoder(
            'navteca/quora-roberta-base',
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.passage_ranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        # self.system_role = args.get("system_role", "You are a student full of curiosity")
        self.qgen_system_role = "You are a student full of curiosity"
        self.n_questions = args.get("n_questions", 5)
        self.question_gen_round = args.get("question_gen_round", 1)
        self.qgen_temp = args.get("qgen_temp", 0.7)
        self.search_timeout = args.get("search_timeout", 10)
        self.max_search_results_per_query = args.get("max_search_results_per_query", 5)
        self.max_passages_per_search_result_to_return = args.get("max_passages_per_search_result_to_return", 3)
        self.sentences_per_passage = args.get("sentences_per_passage", 5)
        self.max_passages_per_question = args.get("max_passages_per_question", 5)
        self.max_aggregated_evidences = args.get("max_aggregated_evidences", 5)
        self.question_persist_path = args.get("question_persist_path", 'questions.txt')
        self.snippets_persist_path = args.get("snippets_persist_path", "passage.json")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        claims_with_evidences = {}
        for i, claim in enumerate(claims):
            evidences = self.get_web_evidences_for_claim(claim)
            claims_with_evidences[claim] = [x['text'] for x in evidences['aggregated']]
        state.set(self.output_name, claims_with_evidences)
        return True, state

    def generate_questions(self, claim, max_loop=5):
        questions = []
        while len(questions) <= 0:
            questions = self.run_question_generation(claim)
            if len(questions) >= 0:
                questions = self.remove_duplicate_questions(questions)
        save_txt(questions, self.question_persist_path)
        return questions

    def retrieve_documents(self, questions):
        snippets = {}
        for question in questions:
            retrieved_passages = self.get_relevant_snippets(question)
            snippets[question] = sorted(
                retrieved_passages,
                key=lambda x: x['retrieval_score'],
                reverse=True
            )[:self.max_passages_per_question]
        save_json(snippets, self.snippets_persist_path)
        return snippets

    def get_web_evidences_for_claim(self, claim):
        evidences = dict()
        evidences["aggregated"] = list()
        questions = self.generate_questions(claim)
        snippets = self.retrieve_documents(questions)
        evidences["question_wise"] = snippets
        total_snippets = sum(list(map(lambda x: len(x), snippets.values())))
        if total_snippets == 0:
            raise RuntimeError("No passages are retrieved, check your network...")
        if total_snippets > self.max_aggregated_evidences:
            while len(evidences["aggregated"]) < self.max_aggregated_evidences:
                for key in evidences["question_wise"]:
                    # Take top evidences for each question
                    if len(evidences["question_wise"][key]) > 0:
                        index = int(len(evidences["aggregated"]) / len(evidences["question_wise"]))
                        evidence = evidences["question_wise"][key][index]
                        evidences["aggregated"].append(evidence)
        else:
            evidences["aggregated"] = itertools.chain.from_iterable(list(snippets.values()))
        return evidences

    @backoff.on_exception(backoff.expo, RateLimitError)
    def run_question_generation(self, claim):
        questions = set()
        for _ in range(self.question_gen_round):
            user_input = QGEN_PROMPT_FMT.format(claim=claim, n=self.n_questions)
            response = gpt(
                user_input,
                model=self.model,
                system_role=self.qgen_system_role,
                num_retries=self.num_retries,
                temperature=self.qgen_temp
            )
            try:
                cur_round_questions = set(eval(response))
                questions.update(cur_round_questions)
            except Exception as e:
                print(f"An unexpected error occurred: {e}.")
        questions = list(sorted(questions))
        return questions

    def remove_duplicate_questions(self, all_questions):
        qset = [all_questions[0]]
        for question in all_questions[1:]:
            q_list = [(q, question) for q in qset]
            scores = self.question_duplicate_model.predict(q_list)
            if np.max(scores) < 0.60:
                qset.append(question)
        return qset

    def scrape_url(self, url: str, timeout: float = 3) -> Tuple[str, str]:
        """Scrapes a URL for all text information.

        Args:
            url: URL of webpage to scrape.
            timeout: Timeout of the requests call.
        Returns:
            web_text: The visible text of the scraped URL.
            url: URL input.
        """
        # Scrape the URL
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as _:
            print("URL Error", url)
            return None, url

        # Extract out all text from the tags
        try:
            soup = bs4.BeautifulSoup(response.text, "html.parser")
            texts = soup.findAll(text=True)
            # Filter out invisible text from the page.
            visible_text = filter(self.is_tag_visible, texts)
        except Exception as _:
            print("Parsing Error", response.text)
            return None, url

        # Returns all the text concatenated as a string.
        web_text = " ".join(t.strip() for t in visible_text).strip()
        # Clean up spacing.
        web_text = " ".join(web_text.split())
        return web_text, url

    def is_tag_visible(self, element: bs4.element) -> bool:
        """Determines if an HTML element is visible.

        Args:
            element: A BeautifulSoup element to check the visiblity of.
        returns:
            Whether the element is visible.
        """
        if element.parent.name in [
            "style",
            "script",
            "head",
            "title",
            "meta",
            "[document]",
        ] or isinstance(element, bs4.element.Comment):
            return False
        return True

    def search_google(self, query: str, num_web_pages: int = 10, timeout: int = 6, save_url: str = '') -> List[str]:
        """Searches the query using Google. 
        Args:
            query: Search query.
            num_web_pages: the number of web pages to request.
            save_url: path to save returned urls, such as 'urls.txt'
        Returns:
            search_results: A list of the top URLs relevant to the query.
        """
        query = query.replace(" ", "+")

        # set headers: Google returns different web-pages according to agent device
        # desktop user-agent
        USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
        # mobile user-agent
        MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; Android 7.0; SM-G930V Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 Mobile Safari/537.36"
        headers = {'User-Agent': USER_AGENT}

        # set language
        # set the Google interface language, use &hl=XX
        # set the preferred language of the search results, use &lr=lang_XX
        # set language as en, otherwise it will return many translation web pages to Arabic that can't be opened correctly.
        lang = "en"

        # scrape google results
        urls = []
        for page in range(0, num_web_pages, 10):
            # here page is google search's bottom page meaning, click 2 -> start=10
            # url = "https://www.google.com/search?q={}&start={}".format(query, page)
            url = "https://www.google.com/search?q={}&lr=lang_{}&hl={}&start={}".format(query, lang, lang, page)
            r = requests.get(url, headers=headers, timeout=timeout)
            # collect all urls by regular expression
            # how to do if I just want to have the returned top-k pages?
            urls += re.findall('href="(https?://.*?)"', r.text)

        # set to remove repeated urls
        urls = list(set(urls))

        # save all url into a txt file
        if not save_url == "":
            with open(save_url, 'w') as file:
                for url in urls:
                    file.write(url + '\n')
        return urls

    def chunk_text(
            self,
            text: str,
            tokenizer,
            sentences_per_passage: int = 5,
            filter_sentence_len: int = 250,
            sliding_distance: int = 2,
    ) -> List[str]:
        """Chunks text into passages using a sliding window.

        Args:
            text: Text to chunk into passages.
            sentences_per_passage: Number of sentences for each passage.
            filter_sentence_len: Maximum number of chars of each sentence before being filtered.
            sliding_distance: Sliding distance over the text. Allows the passages to have
                overlap. The sliding distance cannot be greater than the window size.
        Returns:
            passages: Chunked passages from the text.
        """
        if not sliding_distance or sliding_distance > sentences_per_passage:
            sliding_distance = sentences_per_passage
        assert sentences_per_passage > 0 and sliding_distance > 0

        passages = []
        try:
            doc = tokenizer(text[:500000])  # Take 500k chars to not break tokenization.
            sents = [
                s.text.replace("\n", " ")
                for s in doc.sents
                if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
            ]
            for idx in range(0, len(sents), sliding_distance):
                passages.append(
                    (" ".join(sents[idx: idx + sentences_per_passage]), idx, idx + sentences_per_passage - 1))
        except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
            print("Unicode error when using Spacy. Skipping text.")

        return passages

    def get_relevant_snippets(
            self,
            query,
    ):
        search_results = self.search_google(query, timeout=self.search_timeout)

        with concurrent.futures.ThreadPoolExecutor() as e:
            scraped_results = e.map(self.scrape_url, search_results, itertools.repeat(self.search_timeout))
        # Remove URLs if we weren't able to scrape anything or if they are a PDF.
        scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]
        # print("Num Bing Search Results: ", len(scraped_results))
        retrieved_passages = list()
        for webtext, url in scraped_results[:self.max_search_results_per_query]:
            passages = self.chunk_text(
                text=webtext,
                tokenizer=self.tokenizer,
                sentences_per_passage=self.sentences_per_passage
            )
            if not passages:
                continue

            # Score the passages by relevance to the query using a cross-encoder.
            scores = self.passage_ranker.predict([(query, p[0]) for p in passages]).tolist()
            # Take the top passages_per_search passages for the current search result.
            passage_scores = sorted(zip(passages, scores), reverse=True, key=lambda x: x[1])

            relevant_items = list()
            for passage_item, score in passage_scores:
                overlap = False
                if len(relevant_items) > 0:
                    for item in relevant_items:
                        if passage_item[1] >= item[1] and passage_item[1] <= item[2]:
                            overlap = True
                            break
                        if passage_item[2] >= item[1] and passage_item[2] <= item[2]:
                            overlap = True
                            break

                # Only consider top non-overlapping relevant passages to maximise for information 
                if not overlap:
                    relevant_items.append(deepcopy(passage_item))
                    retrieved_passages.append(
                        {
                            "text": passage_item[0],
                            "url": url,
                            "sents_per_passage": self.sentences_per_passage,
                            "retrieval_score": score,  # Cross-encoder score as retr score
                        }
                    )
                if len(relevant_items) >= self.max_passages_per_search_result_to_return:
                    break
        # print("Total snippets extracted: ", len(retrieved_passages))
        return retrieved_passages
