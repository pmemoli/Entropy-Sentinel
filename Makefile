PYTHON ?= uv run python

run-stem:
	bash src/scripts/run_stem_scenarios.sh

judge-stem:
	$(PYTHON) -m src.scripts.judge_stem_runs

features-stem:
	$(PYTHON) -m src.scripts.generate_stem_features

classifiers-stem:
	$(PYTHON) -m src.scripts.train_stem_classifiers

audit-stem:
	$(PYTHON) -m src.scripts.audit_stem $(AUDIT_ARGS)

sensitivity-stem:
	bash src/scripts/stem_sensitivity_sweep.sh

run-monitoring:
	bash src/scripts/run_monitoring_scenarios.sh

judge-monitoring:
	$(PYTHON) -m src.scripts.judge_monitoring_runs

features-monitoring:
	$(PYTHON) -m src.scripts.generate_monitoring_features

classifiers-monitoring:
	$(PYTHON) -m src.scripts.train_monitoring_classifiers

pipeline-stem: run-stem judge-stem features-stem

pipeline-monitoring: run-monitoring judge-monitoring features-monitoring

push:
	bash sync/sync.sh

pull:
	bash sync/download.sh

ssh:
	bash sync/ssh.sh
