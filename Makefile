.PHONY: help install test lint run-api run-dashboard docker-up docker-down clean

help:
	@echo "Comandos disponiveis:"
	@echo "  install       - Instala dependencias"
	@echo "  test          - Executa testes"
	@echo "  lint          - Executa linter"
	@echo "  run-api       - Inicia API Flask"
	@echo "  run-dashboard - Inicia dashboard Streamlit"
	@echo "  docker-up     - Sobe containers Docker"
	@echo "  docker-down   - Para containers Docker"
	@echo "  clean         - Remove artefatos de build"

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

run-api:
	PYTHONPATH=. python -m flask --app src.api.app run --port 5000 --debug

run-dashboard:
	PYTHONPATH=. streamlit run src/visualization/dashboard.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
