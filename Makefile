.PHONY: install lint test train evaluate run docker clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

run:
	uvicorn hotel_recommender.api:app --reload

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info artifacts reports
