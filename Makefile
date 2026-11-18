# ARCA · atajos. Escribe `make` sin argumentos para ver la lista.

.DEFAULT_GOAL := help
UV ?= uv

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Instala todo en .venv con uv (fase 1 + RAG + agente + rastreador + dev)
	$(UV) sync --extra train --extra rag --extra agent --extra crawler

setup-min: ## Instala solo lo necesario para la API y los tests
	$(UV) sync

lock: ## Regenera uv.lock tras cambiar pyproject.toml
	$(UV) lock

test: ## Ejecuta los tests
	$(UV) run pytest

lint: ## Comprueba estilo con ruff
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Formatea el código con ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

check-gpu: ## Diagnóstico de GPU y librerías
	$(UV) run arca-check-gpu

smoke: ## Entrenamiento GRPO de prueba en GPU (10-15 min)
	$(UV) run arca-smoke

smoke-dry: ## Prueba de instalación en CPU con un modelo diminuto (~1 min)
	$(UV) run arca-smoke --dry-run

# Fase 1: codificación MedDRA. Pasos y justificación en rlm/MEDDRA.md. Los valores son los de la
# versión final (Qwen3-1.7B, DGX); RLM_MODEL=Qwen/Qwen3-0.6B reproduce la primera versión.
RLM_MODEL ?= Qwen/Qwen3-1.7B
RLM_TEACHER ?= Qwen/Qwen3-4B
RLM_W ?= rlm/weights
RLM_GRPO_ARGS = --data rlm/data/train.jsonl --model $(RLM_MODEL) --init-adapter $(RLM_W)/sft_lora \
	--steps 100 --max-completion-length 1024 --batch-size 1 --save-steps 25
RLM_EVAL_ARGS = --data rlm/data/test.jsonl --model $(RLM_MODEL) --max-new-tokens 2048

rlm-data: ## Fase 1: genera train, test y test_ood (semilla fija) y sus estadísticas
	$(UV) run python -m rlm.generate_meddra --stats reports/phase1_dataset_stats.json

rlm-distill: ## Fase 1: trazas del profesor filtradas por el verificador MedDRA (GPU)
	$(UV) run python -m rlm.distill --data rlm/data/train.jsonl --teacher $(RLM_TEACHER) \
		--samples 4 --n-problems 400 --batch-size 2 --max-new-tokens 1536 \
		--output rlm/data/sft_traces.jsonl

rlm-sft: ## Fase 1: SFT con LoRA sobre las trazas verificadas (GPU)
	$(UV) run python -u -m rlm.train_sft --data rlm/data/sft_traces.jsonl --model $(RLM_MODEL) \
		--output $(RLM_W)/sft_lora

rlm-grpo: ## Fase 1: GRPO desde el adaptador de SFT con las tres recompensas (GPU, ~1 h)
	$(UV) run python -u -m rlm.train_grpo $(RLM_GRPO_ARGS) --output $(RLM_W)/final_rlm_lora

rlm-eval: ## Fase 1: pass@1 base / SFT / GRPO en test y OOD, y curvas de entrenamiento (GPU)
	$(UV) run python -u -m rlm.evaluate $(RLM_EVAL_ARGS) --out reports/phase1_eval_1.7b.json \
		--adapters base=none sft=$(RLM_W)/sft_lora grpo=$(RLM_W)/final_rlm_lora \
		--history $(RLM_W)/sft_lora/trainer_state.json $(RLM_W)/final_rlm_lora/trainer_state.json
	$(UV) run python -u -m rlm.evaluate --data rlm/data/test_ood.jsonl --model $(RLM_MODEL) \
		--max-new-tokens 2048 --n-examples 100 --out reports/phase1_eval_ood_1.7b.json \
		--adapters base=none sft=$(RLM_W)/sft_lora grpo=$(RLM_W)/final_rlm_lora

rlm-ablations: ## Fase 1: GRPO sin recompensa de dominio y con crédito parcial, y su evaluación (GPU)
	$(UV) run python -u -m rlm.train_grpo $(RLM_GRPO_ARGS) --output $(RLM_W)/grpo_nodomain \
		--reward-weights 0.5 2.0 0
	$(UV) run python -u -m rlm.evaluate $(RLM_EVAL_ARGS) --adapters grpo_nodomain=$(RLM_W)/grpo_nodomain \
		--out reports/phase1_eval_ablation_nodomain_1.7b.json
	$(UV) run python -u -m rlm.train_grpo $(RLM_GRPO_ARGS) --output $(RLM_W)/grpo_partial --partial-credit
	$(UV) run python -u -m rlm.evaluate $(RLM_EVAL_ARGS) --adapters grpo_partial=$(RLM_W)/grpo_partial \
		--out reports/phase1_eval_ablation_partial_1.7b.json

# Fase 2: herramientas
tools-eval: ## Fase 2: banco de 30 casos con y sin herramientas (GPU)
	$(UV) run python -m tool_use.evaluate --out reports/phase2_eval.json

# Fase 3: RAG sobre fichas técnicas (CIMA) y guías de la EMA. Detalle en rag/CORPUS.md
rag-corpus: ## Fase 3: descarga el corpus (fichas técnicas de CIMA y guías de la EMA)
	$(UV) run python -m rag.download_corpus

rag-gold: ## Fase 3: comprueba que todas las citas del conjunto dorado existen en el corpus
	$(UV) run python -m rag.gold rag/gold/gold.jsonl

rag-eval: ## Fase 3: chunking, tres retrievers, barrido de λ, instrucciones y contexto (GPU)
	$(UV) run python -m rag.evaluate --gold rag/gold/gold.jsonl

rag-eval-gen: ## Fase 3: lo anterior más la generación con citas sobre el conjunto dorado (GPU)
	$(UV) run python -m rag.evaluate --gold rag/gold/gold.jsonl --generation

rag-test: ## Fase 3: mide UNA vez el conjunto de test (100 preguntas) con la configuración fijada (GPU)
	$(UV) run python -m rag.evaluate --gold rag/gold/gold_test.jsonl --strategies recursive 		--fixed-lambda 0.3 --generation --out reports/phase3_test.json

# Fase 4: agente. Un proceso por cerebro (en GPUs de 8 GB no caben dos a la vez)
agent-eval: ## Fase 4: los tres cerebros ReAct y el agente de código sobre las 20 tareas (GPU)
	for b in base rlm thinking code-thinking code-base; do \
		$(UV) run python -u -m agent.evaluate --brains $$b --out reports/phase4_eval_1.7b.json; \
	done

# Rastreador de Bluesky (proyecto final). Detalle en crawler/README.md
CRAWLER_BRAIN ?= 27b
crawler-marcas: ## Rastreador: marcas comerciales de cada fármaco desde CIMA
	$(UV) run python -m crawler.cima --refresh

crawler-muestra: ## Rastreador: muestra congelada de Bluesky (reanudable; sin GPU)
	$(UV) run --extra crawler python -m crawler.collect

crawler-lote: ## Rastreador: hilos a anotar (lote.csv) y muestra para la doble anotación ciega
	$(UV) run python -m crawler.annotate seleccionar

crawler-prenotar: ## Rastreador: pre-anotación del lote con el 27B (DGX)
	$(UV) run python -m crawler.annotate prenotar --brain $(CRAWLER_BRAIN)

crawler-gold: ## Rastreador: lote anotado → <datos>/anotacion/gold.jsonl
	$(UV) run python -m crawler.annotate consolidar $(or $(ARCA_CRAWLER_DATA),data/bluesky)/anotacion/lote.csv

crawler-eval: ## Rastreador: fija, fija+hilos, híbrido y agente sobre la muestra (DGX)
	$(UV) run python -m crawler.evaluate --systems fija fija_hilos
	$(UV) run python -m crawler.evaluate --systems hibrido agente --brain $(CRAWLER_BRAIN)

crawler-vigilar: ## Rastreador: una ejecución del vigilante periódico (añade --bucle para repetir)
	$(UV) run --extra crawler python -m crawler.monitor --brain $(CRAWLER_BRAIN)

api: ## Levanta la API en local con recarga automática
	$(UV) run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

enunciado: ## Recompila docs/enunciado.pdf a partir del .tex
	cd docs && pdflatex -interaction=nonstopmode enunciado.tex >/dev/null
	cd docs && pdflatex -interaction=nonstopmode enunciado.tex >/dev/null
	cd docs && rm -f enunciado.aux enunciado.log enunciado.out
	@echo "docs/enunciado.pdf actualizado"

build: ## Construye la imagen Docker
	docker compose build

docker-check-gpu: ## Diagnóstico de GPU dentro del contenedor
	docker compose run --rm check-gpu

docker-smoke: ## Smoke test dentro del contenedor
	docker compose run --rm smoke

docker-api: ## API dentro del contenedor
	docker compose up api

shell: ## Shell interactiva con GPU dentro del contenedor
	docker compose run --rm train

.PHONY: help setup setup-min lock test lint format check-gpu smoke smoke-dry rlm-data rlm-distill rlm-sft rlm-grpo rlm-eval rag-corpus rag-gold rag-eval rag-eval-gen rag-test crawler-marcas crawler-muestra crawler-lote crawler-prenotar crawler-gold crawler-eval crawler-vigilar api enunciado build docker-check-gpu docker-smoke docker-api shell
