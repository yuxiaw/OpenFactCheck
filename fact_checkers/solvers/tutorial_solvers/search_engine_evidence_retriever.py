from .utils.api import chatgpt, search_google, search_bing
import openai
import time
from .utils.prompt_base import QGEN_PROMPT
from typing import List, Dict, Any
from .utils.web_util import scrape_url, select_doc_by_keyword_coverage, select_passages_by_semantic_similarity
import json
from core import register_solver
from core.fact_check_state import FactCheckerState
from core.task_solver import StandardTaskSolver


@register_solver("search_engine_evidence_retriever", "claims", "evidences")
class SearchEngineEvidenceRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.search_engine = args.get("search_engine", "google")
        self.search_engine_func = {
            "google": search_google,
            "bing": search_bing
        }.get(self.search_engine, "google")

        self.url_merge_method = args.get("url_merge_method", "union")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        queries = self.generate_questions_as_query(claims)
        evidences = self.search_evidence(claims, queries)
        state.set(self.output_name, evidences)
        return True, state

    # generate questions and queries based on a claim
    def generate_questions_as_query(self, claims,
                                    num_retries: int = 3) -> List[list]:
        """
        num_retries: the number of retries when error occurs during openai api calling
        """
        query_list = []
        for i, claim in enumerate(claims):
            for _ in range(num_retries):
                try:
                    response = chatgpt(QGEN_PROMPT + claim)
                    break
                except openai.OpenAIError as exception:
                    print(f"{exception}. Retrying...")
                    time.sleep(1)
            query_list.append(response)
            # print(response)
            # print("\n")

        # convert openai output: a string into a list of questions/queries
        # not check-worthy claims: query response is set as "", accordingly return a []
        # other responses are split into a list of questions/queries
        automatic_query_list = []
        for query in query_list:
            if query == "":
                automatic_query_list.append([])
            else:
                new_tmp = []
                tmp = query.split("\n")
                for q in tmp:
                    q = q.strip()
                    if q == "" or q == "Output:":
                        continue
                    elif q[:6] == "Output":
                        q = q[7:].strip()
                    new_tmp.append(q)
                automatic_query_list.append(new_tmp)

        return automatic_query_list

    # ----------------------------------------------------------
    # Evidence Retrieval
    # ----------------------------------------------------------
    def collect_claim_url_list(self, queries: List[str]) -> List[str]:
        """
        collect urls for a claim given the query list:
        queries: a list of queries or questions for a claim
        search_engine: use which search engine to retrieve evidence, google or bing
        url_union_or_intersection: url operation, to merge all -> 'union' or obtain intersection
        intersection urls tend to be what is not expected, less relevant
        """
        if len(queries) == 0:
            print("Invalid queries: []")
            return None

        urls_list: List[list] = []  # initial list of urls for all queries
        url_query_dict: Dict[str, list] = {}  # url as key, and list of queries corresponding to this url as value.
        url_union, url_intersection = [], []

        for query in queries:
            urls = self.search_engine_func(query)
            urls_list.append(urls)

        for i, urls in enumerate(urls_list):
            for url in urls:
                if url_query_dict.get(url) is None:
                    url_query_dict[url] = [queries[i]]
                else:
                    url_query_dict[url] = url_query_dict[url] + [queries[i]]

        if self.url_merge_method == "union":
            for urls in urls_list:
                url_union += urls
            url_union = list(set(url_union))
            assert (len(url_union) == len(url_query_dict.keys()))
            return list(url_query_dict.keys()), url_query_dict
        elif self.url_merge_method == "intersection":
            url_intersection = urls_list[0]
            for urls in urls_list[1:]:
                url_intersection = list(set(url_intersection).intersection(set(urls)))
            return url_intersection, url_query_dict
        else:
            print("Invalid url operation, please choose from 'union' and 'intersection'.")
            return None, url_query_dict

    def search_evidence(self,
                        decontextualised_claims: List[str],
                        automatic_query_list: List[list],
                        path_save_evidence: str = "evidence.json",
                        save_web_text: bool = False) -> Dict[str, Dict[str, Any]]:

        assert (len(decontextualised_claims) == len(automatic_query_list))

        claim_info: Dict[str, Dict[str, Any]] = {}
        for i, claim in enumerate(decontextualised_claims):
            queries = automatic_query_list[i]
            if len(queries) == 0:
                claim_info[claim] = {"claim": claim, "automatic_queries": queries, "evidence_list": []}
                print("Claim: {} This is an opinion, not check-worthy.".format(claim))
                continue

            # for each checkworthy claim, first gather urls of related web pages
            urls, url_query_dict = self.collect_claim_url_list(queries)

            docs: List[dict] = []
            for j, url in enumerate(urls):
                web_text, _ = scrape_url(url)
                if not web_text is None:
                    docs.append({"query": url_query_dict[url], "url": url, "web_text": web_text})
                else:
                    continue
            print("Claim: {}\nWe retrieved {} urls, {} web pages are accessible.".format(claim, len(urls), len(docs)))

            # we can directly use the first k of url_query_dict, as it is the list of google returned.
            # Here, we select the most relevent top-k docs against the claim by keyword coverage
            # return index of selected documents as the order in docs
            if len(docs) != 0:
                docs_text = [d['web_text'] for d in docs]
                selected_docs_index = select_doc_by_keyword_coverage(claim, docs_text)
                print(selected_docs_index)
            else:
                # no related web articles collected for this claim, continue to next claim
                claim_info[claim] = {"claim": claim, "automatic_queries": queries, "evidence_list": []}
                continue

            selected_docs = [docs_text[i] for i in selected_docs_index]
            # score corresponding passages and select the top-5 passages
            # return the text of passages; and a list of doc ids for each passage.
            # ids here is as the total number and order in selected_docs_index such as in [4, 25, 28, 32, 33]
            topk_passages, passage_doc_id = select_passages_by_semantic_similarity(claim, selected_docs)

            # recover doc_id to original index in docs which records detailed information of a doc
            passage_doc_index = []
            for ids in passage_doc_id:
                passage_doc_index.append([selected_docs_index[id] for id in ids])

            # evidence list
            evidence_list: List[dict] = []
            for pid, p in enumerate(topk_passages):
                doc_ids = passage_doc_index[pid]
                if save_web_text:
                    evidence_list.append({"evidence_id": pid, "web_page_snippet_manual": p,
                                          "query": [docs[doc_id]["query"] for doc_id in doc_ids],
                                          "url": [docs[doc_id]["url"] for doc_id in doc_ids],
                                          "web_text": [docs[doc_id]["web_text"] for doc_id in doc_ids], })
                else:
                    evidence_list.append({"evidence_id": pid, "web_page_snippet_manual": p,
                                          "query": [docs[doc_id]["query"] for doc_id in doc_ids],
                                          "url": [docs[doc_id]["url"] for doc_id in doc_ids],
                                          "web_text": [], })
            claim_info[claim] = {"claim": claim, "automatic_queries": queries, "evidence_list": evidence_list}

        # write to json file
        # Serializing json
        json_object = json.dumps(claim_info, indent=4)

        # Writing to sample.json
        with open(path_save_evidence, "w") as outfile:
            outfile.write(json_object)
        return claim_info
