import os
import re
import openai
from openai import OpenAI
import requests
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------
# OpenAI ChatGPT and davicci-text
# ----------------------------------------------------------
client = None
def init_client():
    global client
    if client is None:
        if openai.api_key is None and 'OPENAI_API_KEY' not in os.environ:
            print("openai_key not presented, delay to initialize.")
            return
        client = OpenAI()

def chatgpt(user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a NLP expert that is good at fact checking"},
                {"role": "user", "content": user_input},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def davinci(prompt):
    # Set up the model and prompt
    model_engine = "text-davinci-003"

    # Generate a response
    completion = client.completions.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    return response

# ----------------------------------------------------------
# Bing Search
# ----------------------------------------------------------
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search/"
SUBSCRIPTION_KEY = "" # fill your bing api key

def search_bing(query: str, timeout: float = 3) -> List[str]:
    """Searches the query using Bing.
    Args:
        query: Search query.
        timeout: Timeout of the requests call.
    Returns:
        search_results: A list of the top URLs relevant to the query.
    """
    
    headers = {"Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()

    response = response.json()
    search_results = [r["url"] for r in response["webPages"]["value"]]
    return search_results

# Test Bing search 
# search_results = search_bing("What are the different awards that Preslav Nakov has received")
# print(search_results)


# ----------------------------------------------------------
# Google Search
# ----------------------------------------------------------
def search_google(query: str, num_web_pages: int = 10, save_url: str = '') -> List[str]:
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
        r = requests.get(url, headers=headers)
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

# Test google search
# query = "Google Company Introduction"
# urls = search_google(query)
# print(len(urls))
