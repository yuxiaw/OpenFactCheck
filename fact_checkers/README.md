# OpenFactChecking::LLM Fact Checkers

- This work seeks to unify both existing and forthcoming fact-checking solutions within a structured pipeline framework.
- Fact-checking inherently involves a series of sub-tasks, including claim extraction, evidence retrieval, and more.
- We identified that they could be strung into a pipline with specific task solver for each sub-task.
- Each task solver independently manages its logic and resources, yet exposes input and output names to other solvers.
- This framework follows a principle that ensures the system maintains sufficient flexibility to address various potential
expansion requirements through specifications and configurations.
- The framework is deployable with well-functioning frontend and backend components.

## Requirements

- Python 3.8+
- openai
- PyYAML

## Quick Start

To have a quick run, you can simply consider this repo as a python library by importing the core module and run it with
function calling.
Before running it, you need to add your ```openai_key``` in the config file
in ```./config/config.yaml```, which is necessary.
By running the following code segment, you will use our pre-defined pipline to get the revised response when given a
response from an LLM which you don't know whether there are counterfactual statements inside.

```python
from pipeline import Pipeline
from argparse import Namespace

args = Namespace(
    config='./config/config.yaml',
    output='./truth'
)
p = Pipeline(args)
response = "Alan Turing, a British mathematician and computer scientist, is widely regarded as one of the key figures in the development of modern computing."
print(p(response=response))
```

We also provided a command-line entrance:
```shell
python src/llm_fact_checker.py --config ./config/config.yaml --input <input_content> --output ./truth
```
where the ```input_content``` can be a text file containing multiple lines of LLM responses, or can also be a
response string.

## Use as a web service
A simple web service API has been implemented in ```./src/web_service.py```. You can start the backend system by running for following command:
```shell
python src/web_service.py --config ./config/web_service_config.yaml --user_src ./solvers/web_service_solvers/
```
Remember to set the required API keys in ```./config/web_service_config.yaml``` before starting the server.

## Advanced Usage

If you want to have a more customized pipline with our code base, we recommend you to have a look on the configuration
file in ```./config/config.yaml```.
In this file, we pre-defined a pipline with several task solvers listed in the ```solvers``` entry (8 solvers in our
pipeline).
Each solver has their name and configurations represented in a key-value format.
You may notice that, arguments for each solver can be different, but all of them have the ```input_name``` and ```output_name``` fields, and the ```output_name``` name should be exactly same to the ```input_name``` of next solver.
The reason of doing so is that we want to provide flexibility to the framework, so that you can plug new solvers or remove existing solvers simply by re-connecting them with correct input and output name.
Let's take an example, if we don't want the pipline to perform post editing on the examined claims, we can define a pipline like this:
```yaml
solvers:
  all_pass_abstain_detector:
    input_name: response
    output_name: response
  spacy_response_decomposer:
    spacy_model: en_core_web_sm
    input_name: response
    output_name: sentences
  chatgpt_decontextualizer:
    input_name: sentences
    output_name: claims
  chatgpt_worthiness_filter:
    input_name: claims
    output_name: claims
  search_engine_evidence_retriever:
    search_engine: google
    url_merge_method: union
    input_name: claims
    output_name: evidences
  chat_gpt_claim_examiner:
    path_save_stance: evidence_stance.json
    num_retries: 3
    input_name: evidences
    output_name: output
```
where we remove the last two solvers and return the examined claims as the output.

In summary, through manipulating the config file, you can create your own pipline as long as the solver can correctly process the output from previous solver.


## Extending This Framework
For developers who want to replace some of the solvers with your own modules or build an entire new pipline, we provide a simple tutorial of developing a simple dummy pipeline. All the codes of the tutorial are placed at the `solvers` folder.

In general, there are 5 things you need to do to develop your own fact-checking pipeline.
1. Understand what sub-tasks your pipeline contains.
2. Design a solver for each sub-task, including the input output format and resources it will manage.
3. Implement your solvers.
4. Write your own pipeline config.
5. Run the system with your code and config file.

### Step 1 & 2
Since we are trying to build a dummy pipeline, we don't really need it to be useful, so we first design three useless tasks to simulate a real fact-checker:
1. Fake Claim Extractor:
   - We take a response string, randomly cut it into several sub-strings to mimic the claims extracted from the response.
2. Ignorant Search Engine Retriever
   - We randomly generate some documents for each claim to mimic search engine retrieving.
3. Confused Claim Examiner
   - We randomly label a True/False tag to each claim to represent whether it is a fact.
4. Useless Response Re-Generator 
   - We re-construct a response where counter-factual information are corrected.

### Step 3 & 4
Please see ```./solvers/dummy_solvers``` for the implementation of these solvers, and the config file used by them.

### Step 5
To run the code, you can use both the library way or the command line way. Here we show the library way:
```python
from pipeline import Pipeline
from argparse import Namespace

args = Namespace(
    user_src='./solvers/dummy_solvers',
    config='./solvers/dummy_config.yaml',
    output='./truth'
)
p = Pipeline(args)
response = "Alan Turing, a British mathematician and computer scientist, is widely regarded as one of the key figures in the development of modern computing."
print(p(response=response))
```
Note that for customized solvers developed by external developers, our framework can automatically load your modules as long as you provide the ```user_src```.
