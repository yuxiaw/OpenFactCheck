# Factool integration into the LLM Fact Checker DEMO

## Advanced Usage

The **Factool** integration process follows the guidelines and the architecture of the **llm-fact-checker**. The idea followed by the current implementation, is to maximize the compatibility between the newly developed solvers and the ones present in **llm_fact_checker** code itself. Default *evidence* JSON files are produced in the same format and with the same default names. The I/O interfaces in the solvers in both monolith (blackbox) and micro-service implementations, are 100% compatible with the ones of their **GPT** integration (default **llm_fact_checker**) counterparts. The only difference is the *path_save_analysis* parameter in the **factool_blackbox_post_editor** solver, which saves the **Factool** output of the blackbox (monolith) solver to a JSON file.<br />
Example pipeline has been deployed at ```factool_config.yaml```. The **Factool** blackbox (monolith) integration is guided by ```factool_blackbox_config.yaml```.<br />
A pipeline with micro-service **Factool** setting:
```yaml
openai_key:
serper_key:
scraper_key:
solvers:
  all_pass_abstain_detector:
    input_name: response
    output_name: response
  factool_decontextualizer:
    llm_in_use: gpt-4
    input_name: response
    output_name: claims
  factool_evidence_retriever:
    llm_in_use: gpt-4
    input_name: claims
    output_name: evidences
  factool_claim_examiner:
    llm_in_use: gpt-4
    input_name: evidences
    output_name: claim_info
  factool_post_editor:
    input_name: claim_info
    output_name: claim_info
  concat_response_generator:
    input_name: claim_info
    output_name: output
```
Here, the **[OpenAI](https://beta.openai.com/)**, **[Serper](https://serper.dev/)** and **[Scraper](https://www.scraperapi.com/)** API keys are mandatory for the proper functioning of the **Factool** class. Solvers are identical with the well-known solvers from the **GPT** integration. The *llm_in_use parameter* represents the **OpenAI** LLM currently being employed by the **Factool** components.<br />
The pipeline for the blackbox (monolith) **Factool** is similar, but with less inherent dynamics, employing the **Factool** *class*, instead of it's logically separated components:
```yaml
openai_key:
serper_key:
scraper_key:
solvers:
  all_pass_abstain_detector:
    input_name: response
    output_name: response
  factool_blackbox:
    llm_in_use: gpt-4
    input_prompt: question
    input_name: response
    output_name: claim_info
  factool_blackbox_post_editor:
    path_save_analysis: factool_evidence_analysis.json
    input_name: claim_info
    output_name: claim_info
  concat_response_generator:
    input_name: claim_info
    output_name: output
```

## Example

The following example code encompases the execution of the Factool micro-services pipeline:
```python
from pipeline import Pipeline
from argparse import Namespace

args = Namespace(
    user_src='../src/solvers',
    config='../config/factool_config.yaml',
    output='./truth'
)
p = Pipeline(args)
question = "Who is Alan Turing?"
response = "Alan Turing used to be Serbian authoritarian leader, mathematician and computer scientist. He used to be a leader of the French Resistance."
print(p(question=question, response=response))
```
