import bs4
import spacy
import requests
from collections import Counter
from string import punctuation
from typing import List, Dict, Tuple, Any


def is_tag_visible(element: bs4.element) -> bool:
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


def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
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
        print("URL Require Error.")
        return None, url

    # Extract out all text from the tags
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        texts = soup.findAll(text=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:
        print("BS4 Error.")
        return None, url

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url


def get_hotwords(text: str, top_k: int = 10) -> List[str]:
    """# extract key words for a text, return most frequent topk keywords
    """
    nlp = spacy.load("en_core_web_sm")
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower()) 

    result = []
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
            
    most_common_list = Counter(result).most_common(top_k)
    keywords = [item[0] for item in most_common_list]
    return keywords


def select_doc_by_keyword_coverage(claim: str, docs: List[str], 
                                   top_k_keywords: int = 10, top_k_docs: int = 5) -> List[int]:
    """count how many keywords appeared in this document len(appeared_keywords)
       sort documents by the count that represents the degree of coverage of the claim for the doc
       return index of top-k docs"""
    # get keywords in the claim.
    keywords = get_hotwords(claim, top_k_keywords)
    
    # how many keywords are contained in each doc
    counts = []
    for doc in docs:
        doc = doc.lower() # as all keywords are lowercase
        count = [1 for word in keywords if word in doc]
        counts.append(sum(count))

    # we keep the docs that contain the most keywords, as we aim to cut off lots of unrelevant docs
    max_count = max(counts)
    selected_docs_index = [i for i in range(len(docs)) if counts[i] == max_count]
    if len(selected_docs_index) < top_k_docs:
        # we sort docs by coverage, then keep top-K
        docs_index_sorted_coverage = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
        selected_docs_index = docs_index_sorted_coverage[:top_k_docs]
    
    print("There are {} web pages selected.".format(len(selected_docs_index)))
    return selected_docs_index


def chunk_text(text: str, sentences_per_passage: int, 
               filter_sentence_len: int, sliding_distance: int = None) -> List[str]:
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
    TOKENIZER = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    assert sentences_per_passage > 0 and sliding_distance > 0

    passages = []
    try:
        doc = TOKENIZER(text[:500000])  # Take 500k chars to not break tokenization.
        sents = [
            s.text
            for s in doc.sents
            if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
        ]
        for idx in range(0, len(sents), sliding_distance):
            passages.append(" ".join(sents[idx : idx + sentences_per_passage]))
    except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
        print("Unicode error when using Spacy. Skipping text.")

    return passages


def select_passages_by_semantic_similarity(claim: str, selected_docs: List[str],
                                           max_sentences_per_passage: int = 3, filter_sentence_len: int = 250,
                                           sliding_distance: int = 3, top_k_passage: int = 5) -> Tuple[list, list]:
    passages: List[str] = []
    for doc in selected_docs:
        # RARR default setting (5, 250, 1) for chunk
        snippets = chunk_text(doc, max_sentences_per_passage, filter_sentence_len, sliding_distance) 
        passages.extend(snippets)
    passages = list(set(passages)) # remove repeated ones
    print("{} snippets of text are splitted.".format(len(passages)))
    
    # score each snippet of text against claim
    nlp = spacy.load("en_core_web_sm")
    claim = nlp(claim)
    sim = []
    for p in passages:
        sim.append(claim.similarity(nlp(p)))
    
    # sort by similarity score and keep topk
    index_sorted_sim = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)
    topk_passages = [passages[i] for i in index_sorted_sim[:top_k_passage]]

    # find docs of topk_passages: one passage may occur in multiple docs
    passage_doc_id: List[list] = []
    for p in topk_passages:
        temp = []
        for id, doc in enumerate(selected_docs):
            if p in doc:
                temp.append(id)
        # if fail to find docs of this passage, just pass.
        # this will lead some [], [], [] in evidence list for this snippet of text 
        if len(temp) == 0:
            print("Error in matching selected passage to its docs!")
        passage_doc_id.append(temp)

    return topk_passages, passage_doc_id