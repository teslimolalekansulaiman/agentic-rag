.PHONY: up ingest api eval test

up:
	docker compose -f infra/docker-compose.yml up -d
	@echo "Waiting for Postgres..." && sleep 5

ingest:
	python -m src.ingestion.cli all data/raw

api:
	uvicorn src.app.api:app --reload --port 8000

eval:
	python -m src.evaluation.ragas_runner

test:
	pytest -q
