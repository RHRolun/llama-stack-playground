#!/bin/bash

cd "$(dirname "$0")"

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run llama_stack/distribution/ui/app.py