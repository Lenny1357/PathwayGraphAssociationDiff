# Common entry points. `make help` lists them.
PY ?= python3

.PHONY: help install test test-offline lint train-ecoli compare score figures data-ecoli data-gem clean

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  %-14s %s\n", $$1, $$2}'

install:        ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:           ## unit tests (fast, CPU, no data)
	$(PY) -m pytest tests -q

test-offline:   ## the data builders' self-checks (no network)
	$(PY) -m pytest tests/test_offline_scripts.py -q -m "not slow"

lint:           ## ruff over the library, scripts and tests
	ruff check graphpath experiments scripts tests

train-ecoli:    ## short smoke run of one arm on E. coli
	$(PY) scripts/train.py --dataset ecoli --model masked --steps 200

compare:        ## the four-arm comparison (PRESET=sept|typed|oct2, DATASETS="ecoli human")
	$(PY) scripts/train_compare.py --preset $(or $(PRESET),sept) --datasets $(or $(DATASETS),ecoli human)

score:          ## score + analyse a run (RUN=<name> or latest:<prefix>)
	$(PY) scripts/score_run.py --run $(RUN)

figures:        ## redraw every figure of a run from its JSON (RUN=<name>)
	$(PY) scripts/make_figures.py --run $(RUN)

data-ecoli:     ## rebuild the E. coli reference, pathways and co-expression
	cd data/ecoli && $(PY) build_reference.py --model iML1515.json --out-dir ecoli_ref && $(PY) build_pathways.py

data-gem:       ## rebuild the stoichiometric models (Human-GEM + iML1515)
	cd data/gem && $(PY) build_gem.py && $(PY) build_gem_ecoli.py

clean:          ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .DS_Store -delete
