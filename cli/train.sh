EPOCHS="${1:-120}"
CHECKPOINT_URL="${2:-None}"

echo 'EPOCHS' $EPOCHS
echo 'CHECKPOINT_URL' $CHECKPOINT_URL

papermill  trainsets/retrain_contract_uber_model.ipynb training_reports/retrain_contract_uber_model-result.ipynb --log-output -p TRAIN_FROM_CP True -p TRAIN_TEST_SPLIT_SEED 43 -p TRAIN True -p TEST_FLOW False -p DEBUG False -p LR 0.0001 -p EPOCHS $EPOCHS -p CHECKPOINT_URL $CHECKPOINT_URL  
