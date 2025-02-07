.PHONY: vertex-run
vertex-run:
	uvicorn vertex.main:app --reload --port 8000

.PHONY: gemini-run
gemini-run:
	uvicorn gemini.main:app --reload --port 8000

.PHONY: check-type
check-type:
	ruff check --fix .

.PHONY: format
format:
	ruff format .
