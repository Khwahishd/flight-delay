.PHONY: setup data train test clean

setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:
	.venv/bin/python -m flightdelay.data

train:
	.venv/bin/python -m flightdelay.train

test:
	.venv/bin/python -m pytest

clean:
	rm -rf data/raw reports/*.json reports/figures/*.png
