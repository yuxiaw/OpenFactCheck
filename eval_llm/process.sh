#!/bin/bash

# uploaded_file_path is the user uploaded response file, which should be a csv file
# An example can be found in eval_llm/model_response/GPT4_response.csv
# the user_id can by any identifier that is used to name the user or task specific folder for storing intermediate results
# The final report is a pdf file stored at ${report_path} (again, at the user specific folder) named report.pdf, which is used for users to download
uploaded_file=$1
user_id=$2

# Validate the input
if [ -z "$uploaded_file" ]; then
  echo "Please provide the path to the uploaded file as the first argument"
  exit 1
fi

if [ -z "$user_id" ]; then
  echo "Please provide the user id as the second argument"
  exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Please provide the OpenAI API key as the environment variable OPENAI_API_KEY"
  exit
fi

# Create the folder structure for storing intermediate results and final report
evaluation_root_path=../eval_results/llm/${user_id}
eval_result_path=${evaluation_root_path}/intermediate_results/
analysis_result_path=${evaluation_root_path}/figure/
report_path=${evaluation_root_path}/report/

mkdir -p ${eval_result_path} ${analysis_result_path} ${report_path}

#Step 1: Evaluate LLM responses
python evaluate_llm_factuality.py --input_path ${uploaded_file} --eval_result_path ${eval_result_path} --openai_apikey ${OPENAI_API_KEY}

if [ $? -ne 0 ]; then
  echo "Failed to evaluate the LLM responses"
  exit 1
fi

#Step 2: Run analysis script
python analyze_results.py --input_path ${eval_result_path} \
   --analysis_result_path ${analysis_result_path} \
   --model_response_path ${uploaded_file}

if [ $? -ne 0 ]; then
  echo "Failed to analyze the evaluation results"
  exit 1
fi

#Step 3: Generate report
python generate_report.py --model_name ${user_id} --input_path ${analysis_result_path}

if [ $? -ne 0 ]; then
  echo "Failed to generate the report"
  exit 1
fi

#Step 4: Copy report to the report path
cp ${analysis_result_path}/main.pdf ${report_path}/report.pdf

if [ $? -ne 0 ]; then
  echo "Failed to copy the report to the report path"
  exit 1
fi

echo "Evaluation completed successfully"