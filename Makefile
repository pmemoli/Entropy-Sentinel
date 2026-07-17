include .env
export

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

sync:
	mutagen sync create --name=entropy-sentinel \
	  --ignore=.venv --ignore=.pytest_cache --ignore=.git --ignore=.teamviewer --ignore=.env \
	  ~/academy/papers/entropy_sentinel \
	  $(HOST):~/Documents/entropy_is_enough

first-sync:
	mutagen sync create --name=entropy-sentinel \
	  --sync-mode=one-way-replica \
	  --ignore=.venv --ignore=.pytest_cache --ignore=.git --ignore=.teamviewer --ignore=.env \
	  ~/academy/papers/entropy_sentinel \
	  $(HOST):~/Documents/entropy_is_enough

terminate-sync:
	mutagen sync terminate entropy-sentinel

ssh:
	ssh $(HOST)

.PHONY: run-stem judge-stem features-stem classifiers-stem audit-stem sensitivity-stem \
		run-monitoring judge-monitoring features-monitoring classifiers-monitoring \
		pipeline-stem pipeline-monitoring sync first-sync terminate-sync
