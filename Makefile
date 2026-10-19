.PHONY: install run debug clean lint lint-strict

UV := $(shell command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")

check-uv:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(UV)" ]; then \
		echo "\033[1;33muv is not installed. Installing...\033[0m"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "\033[1;32muv installed successfully.\033[0m"; \
	fi

install: check-uv
	@$(UV) sync

RUN_ARGS := $(filter-out run --,$(MAKECMDGOALS))

run:
	@$(MAKE) install --no-print-directory
	@uv run python -m src $(RUN_ARGS)
	@$(MAKE) clean --no-print-directory

debug:
	@uv run python -m pdb -m src

clean:
	@rm -rf .mypy_cache .pytest_cache
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -name '*.pyc' -delete

lint:
	@uv run flake8 . --exclude=.venv,data,llm_sdk,__pycache__
	@uv run mypy . --exclude "(.venv|data|llm_sdk)" --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@$(MAKE) clean --no-print-directory

lint-strict:
	@uv run flake8 . --exclude=.venv,data,llm_sdk,__pycache__
	@uv run mypy . --strict --exclude "(.venv|data|llm_sdk)"
	@$(MAKE) clean --no-print-directory

%:
	@:
