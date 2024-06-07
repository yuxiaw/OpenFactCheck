import os
import json
import re
import pandas as pd
os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"
os.environ['SERPAPI_API_KEY'] = "SERPAPI_API_KEY"

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
import faiss
import time
from custom_AutoGPT import CustomAutoGPT

def run_langchain(claim):
    start = time.time()
    prompt = prompt = """Fact check the following claim using evidence from web search. You should make the decision only based on the search results obtained from web search, and not based on any assumptions. DO NOT TRY to scrape the individual URLs obtained via web search. If you fail to get search results, make decision using your own knowledge and output. If you do not find any concrete evidence that directly supports or contradicts the claim, you can search the web the second time, but NOT MORE THAN TWICE. If you do not find any evidence to support the claim after multiple searches, then you can output the claim as false.  OUTPUT ONLY True OR False. Claim: {claim}"""
    
    output = dict()
    output['claim'] = claim
    search = SerpAPIWrapper(serpapi_api_key="SERPAPI_API_KEY")
    tools = [
        Tool(
            name = "search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        )]
    
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    
    # agent = AutoGPT.from_llm_and_tools(
    #     ai_name="Tom",
    #     ai_role="Assistant",
    #     tools=tools,
    #     llm=ChatOpenAI(temperature=0, model_name="gpt-4-turbo-2024-04-09"), #"gpt-3.5-turbo-0125"
    #     memory=vectorstore.as_retriever(),
    # )
    
    agent = CustomAutoGPT.from_llm_and_tools(
        ai_name="Tom",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(temperature=0, model_name="gpt-4-turbo-2024-04-09"), #"gpt-3.5-turbo-0125"
        memory=vectorstore.as_retriever(),
    )
    # Set verbose to be true
    agent.chain.verbose = True
    
    with get_openai_callback() as cb:
        response, loop_count = agent.run([prompt.format(claim=claim)], max_iterations=2)
        # Usually either 'True' or 'False'
        output["langchain_output"] = response
        output["loop_count"] = loop_count
    
    print("total cost: ", str(float(cb.total_cost))) 
    output["langchain_messages"] = list()
    for m in agent.chat_history_memory.messages:
        output["langchain_messages"].append(m.content)
    end = time.time()
    output["usd-cost"] = str(float(cb.total_cost))
    output["time-cost"] = end-start
    
    return output

if __name__ == "__main__":
    
    df = pd.read_json("Factbench.jsonl", lines=True)
    claims, gold_labels = [], []
    for source in ['factool-qa', 'felm-wk']:
        t = df[df['source'] == source]
        for l in t['claim_labels']:
            gold_labels += l
        for c in t['claims']:
            claims += c
    print(len(claims), len(gold_labels))

    outputs = []
    for claim in claims[:2]:
        output = run_langchain(claim)
        outputs.append(output)
        pd.DataFrame(outputs).to_json("langchain_results.jsonl", lines=True, orient="records")