#!/usr/bin/env bash

project_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"

cd $project_directory
# if .venv does not exist, create virtual environment
if [ ! -d .venv ]; then
    virtualenv .venv
fi
# install dependencies
source .venv/bin/activate
pip install --quiet -r requirements.txt

# run project 2 solution
cd p3
time python correlated_q.py -- --verbose
time python foe_q.py -- --verbose
time python friend_q.py -- --verbose
time python q_learning.py -- --verbose
