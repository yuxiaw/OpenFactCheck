#!/bin/bash

source "${BASH_SOURCE%/*}/common.sh"

# Start the frontend
c_echo $GREEN "Starting the frontend"
cd app
streamlit run run.py