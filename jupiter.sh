# shellcheck shell=sh

# activates local jupyter in virtual env
#this allows colab to connect 127.0.0.1

source /Users/artem/work/nemo/goil/nlp_tools/venv/bin/activate

jupyter lab --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0