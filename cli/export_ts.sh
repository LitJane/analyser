limit="${1:-5000}"
papermill trainsets/export_trainset.ipynb training_reports/export_trainset-result.ipynb -p LIMIT $limit --log-output