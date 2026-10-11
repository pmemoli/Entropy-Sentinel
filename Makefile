include .env
export

CUDA_DEVICE_ORDER ?= PCI_BUS_ID
CUDA_VISIBLE_DEVICES ?= 0

PYTHON ?= uv run python

REMOTE_DIR ?= ~/Documents/entropy_is_enough

run-stem:
	bash src/scripts/run_stem_scenarios.sh

regenerate-truncated-stem:
	bash src/scripts/regenerate_truncated_stem.sh

judge-stem:
	$(PYTHON) -m src.scripts.judge_stem_runs $(JUDGE_ARGS)

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

pipeline-stem: run-stem judge-stem features-stem

pipeline-monitoring: run-monitoring judge-monitoring features-monitoring

sync:
	mutagen sync create --name=entropy-sentinel \
	  --sync-mode=two-way-resolved \
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

compile:
	$(MAKE) -C paper compile

clean-paper:
	$(MAKE) -C paper clean

ssh:
	ssh $(HOST)

remote:
	ssh -t $(HOST) 'cd $(REMOTE_DIR) && bash -lc "$(CMD)"'

remote-%:
	ssh -t $(HOST) 'cd $(REMOTE_DIR) && bash -lc "make $*"'

.PHONY: run-stem regenerate-truncated-stem judge-stem features-stem classifiers-stem audit-stem sensitivity-stem \
		run-monitoring judge-monitoring features-monitoring classifiers-monitoring \
		pipeline-stem pipeline-monitoring sync first-sync terminate-sync ssh remote \
		compile clean-paper

ANALYSIS_MODELS = phi3-3b ministral3-3b ministral3-8b qwen3-4b qwen3-8b gemma3-4b gemma3-12b llama3-8b oss-20b

analysis-sweep:
	cd analysis && printf '%s\n' $(ANALYSIS_MODELS) | OMP_NUM_THREADS=1 xargs -P 9 -I{} $(PYTHON) stem_sweep.py {}

analysis-robustness:
	cd analysis && { for m in $(ANALYSIS_MODELS); do echo "bank $$m"; echo "truncation $$m"; echo "seeds $$m 1"; echo "seeds $$m 2"; done; } | OMP_NUM_THREADS=1 xargs -P 9 -L 1 $(PYTHON) robustness.py
	cd analysis && for c in base monotone gbm_deep bag5 rf_deep et lr_poly mlp votes ens_sel ext extnolen agg iso platt; do for m in $(ANALYSIS_MODELS); do echo "search $$c $$m"; done; done | OMP_NUM_THREADS=1 xargs -P 9 -L 1 $(PYTHON) robustness.py

analysis-tables:
	cd analysis && $(PYTHON) section_3_results.py && $(PYTHON) appendix_results.py && $(PYTHON) section_4_results.py && $(PYTHON) section_5_results.py && $(PYTHON) text_numbers.py

analysis: analysis-sweep analysis-robustness analysis-tables

.PHONY: analysis analysis-sweep analysis-robustness analysis-tables
