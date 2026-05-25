PYTHON ?= python3

store:
	bash src/scripts/store_activations.sh

features:
	bash src/scripts/generate_features.sh

evaluate:
	bash src/scripts/evaluate_runs.sh

baselines:
	$(PYTHON) -m src.scripts.train_baselines

classifiers:
	$(PYTHON) -m src.scripts.train_classifiers

audit:
	$(PYTHON) src/scripts/audit.py $(AUDIT_ARGS)

sensitivity:
	bash src/scripts/sensitivity_sweep.sh

train-extremes-intermediate:
	$(PYTHON) -m src.scripts.train_extremes_intermediate

pipeline: store features evaluate

push:
	bash sync/sync.sh

pull:
	bash sync/download.sh

ssh:
	bash sync/ssh.sh
