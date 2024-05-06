#!/bin/bash

source "${BASH_SOURCE%/*}/common.sh"

# Start the backend
c_echo $GREEN "Starting the backend"
cd api
python run.py