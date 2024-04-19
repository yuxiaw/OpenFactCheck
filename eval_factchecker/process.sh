#!/bin/bash

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


# Create the folder structure for storing intermediate results and final report
evaluation_root_path=../eval_results/factchecker/${user_id}

#Step 1: Evaluate factchecker responses
python evaluate_factchecker_factuality.py --input_path ${uploaded_file} --results_path ${evaluation_root_path}