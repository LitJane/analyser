# shellcheck shell=sh

# activates local jupyter in virtual env
source /Users/artem/work/nemo/goil/nlp_tools/venv/bin/activate
jupyter nbconvert --ExecutePreprocessor.timeout=0 --to notebook --execute trainsets/retrain_contract_uber_model.ipynb --output=../../work/retrain_contract_uber_model_results.ipynb
