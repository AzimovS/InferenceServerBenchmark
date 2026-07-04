# ==============================================================================
# Blackwell Inference Testbench — Makefile
# ==============================================================================
#
# PRIMARY WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
#  1. Edit models.yaml — the only file you need to touch.
#  2. Run one of the targets below.
#
#   make sanity LABEL=mistral-7b          Quick 10-request smoke test
#   make concurrency-bench                Goal 1 — rank single-tenant models
#   make concurrency-bench LABEL=qwq-32b  Goal 1 — one model
#   make co-deploy                        Goal 2 — all viable (large, small) pairs
#   make co-deploy LABEL_LARGE=gpt-oss-120b LABEL_SMALL=qwen3-8b  # one pair
#   make probe                            Auto-detect max_model_len for all models
#   make probe LABEL=qwen3-8b             Auto-detect for one model
#
# STT (Speech-to-Text)
# ─────────────────────────────────────────────────────────────────────────────
#   make stt-sanity LABEL=voxtral-mini-4b         Quick STT smoke test
#   make stt-bench LABEL=voxtral-mini-4b          STT concurrency benchmark
#   make stt-streaming-sanity LABEL=voxtral-mini-4b  Streaming STT smoke test
#   make stt-streaming-bench LABEL=voxtral-mini-4b   Streaming STT concurrency benchmark
#   make mixed-co-deploy LABEL_LARGE=gpt-oss-120b LABEL_STT=voxtral-mini-4b
#   make download-stt-data                        Download LibriSpeech test-clean
#
# ONE-OFF / MANUAL
# ─────────────────────────────────────────────────────────────────────────────
#   make serve LABEL=mistral-7b    Boot vLLM for one model (then bench manually)
#   make bench-sanity              Run sanity bench against whatever is up
#   make bench-concurrency         Run concurrency bench against whatever is up
#
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
#   make prefetch                  Download all models in models.yaml to HF cache
#   make logs                      Tail vLLM server logs
#   make status                    Show containers + GPU state
#   make stop                      Stop all containers
#   make results                   List result files (most recent first)
#   make gpu-monitor               One-shot GPU snapshot
# ==============================================================================

DC           := docker compose
PYTHON       := python3
LABEL        ?=   # filter to a single model label (single-model benchmarks)
LABEL_A      ?=   # model on port 8000 (co-serve)
LABEL_B      ?=   # model on port 8001 (co-serve)
LABEL_LARGE  ?=   # filter large-role model (co-deploy)
LABEL_SMALL  ?=   # filter small-role model (co-deploy)
LABEL_STT    ?=   # filter STT model (mixed-co-deploy)
STT_PRIMARY  ?=   # set to 1 to put STT on port 8000 (mixed-co-deploy)

# ==============================================================================
# PRIMARY  —  models.yaml-driven sweeps
# ==============================================================================

sanity:
	$(PYTHON) core/sweep.py --bench sanity $(if $(LABEL),--label $(LABEL),)

concurrency-bench:
	$(PYTHON) core/sweep.py --bench concurrency-bench $(if $(LABEL),--label $(LABEL),)

co-deploy:
	$(PYTHON) core/sweep.py --bench co-deploy \
		$(if $(LABEL_LARGE),--label-large $(LABEL_LARGE),) \
		$(if $(LABEL_SMALL),--label-small $(LABEL_SMALL),)

# Probe: auto-detect actual max_model_len for each model (no --max-model-len passed to vLLM)
probe:
	$(PYTHON) core/sweep.py --probe $(if $(LABEL),--label $(LABEL),)

# Boot a single model without running a benchmark (useful for exploratory runs)
serve:
	@if [ -z "$(LABEL)" ]; then \
		echo "Usage: make serve LABEL=<model-label>"; \
		echo "Labels are defined in models.yaml"; \
		exit 1; \
	fi
	$(PYTHON) core/sweep.py --serve-only --label $(LABEL)

# Boot two models simultaneously: LABEL_A on :8000, LABEL_B on :8001
co-serve:
	@if [ -z "$(LABEL_A)" ] || [ -z "$(LABEL_B)" ]; then \
		echo "Usage: make co-serve LABEL_A=<port-8000-model> LABEL_B=<port-8001-model>"; \
		echo "Labels are defined in models.yaml"; \
		exit 1; \
	fi
	$(PYTHON) core/sweep.py --co-serve $(LABEL_A) $(LABEL_B)

# ==============================================================================
# EMBEDDING sidecar  —  TEI, colocated next to the vLLM servers
# ==============================================================================

# Start the embedding sidecar (default: Qwen/Qwen3-Embedding-4B on :8002).
# Override with EMBED_MODEL / EMBED_PORT / EMBED_MAX_BATCH_TOKENS (see .env.example).
embed-up:
	$(DC) --profile embedding up -d tei-embed

embed-down:
	$(DC) --profile embedding rm -sf tei-embed

embed-logs:
	$(DC) --profile embedding logs -f tei-embed

# Wait for TEI health (first boot loads ~8 GB of weights), then round-trip one
# embedding through the native /embed endpoint and print the vector dimension.
embed-sanity:
	@echo "Waiting for TEI on :$${EMBED_PORT:-8002} ..."
	@for i in $$(seq 1 60); do \
		curl -sf http://localhost:$${EMBED_PORT:-8002}/health > /dev/null && break; \
		if [ $$i -eq 60 ]; then echo "ERROR: TEI not healthy after 5 min — check 'make embed-logs'"; exit 1; fi; \
		sleep 5; \
	done
	@curl -s http://localhost:$${EMBED_PORT:-8002}/embed \
		-H 'Content-Type: application/json' \
		-d '{"inputs": "The quick brown fox jumps over the lazy dog"}' \
		| $(PYTHON) -c 'import sys, json; v = json.load(sys.stdin)[0]; print(f"OK: {len(v)}-dim embedding from TEI")'

# ==============================================================================
# STT (Speech-to-Text)  —  models.yaml-driven sweeps
# ==============================================================================

# Download LibriSpeech test-clean dataset for STT benchmarking
download-stt-data:
	bash assets/download_librispeech.sh

# STT sanity check — quick smoke test
stt-sanity:
	$(PYTHON) core/sweep.py --bench stt-sanity $(if $(LABEL),--label $(LABEL),)

# STT concurrency benchmark — throughput, RTF, WER under concurrent streams
stt-bench:
	$(PYTHON) core/sweep.py --bench stt-concurrency-bench $(if $(LABEL),--label $(LABEL),)

# Mixed co-deploy — text + STT models simultaneously
mixed-co-deploy:
	$(PYTHON) core/sweep.py --bench mixed-co-deploy \
		$(if $(LABEL_LARGE),--label-large $(LABEL_LARGE),) \
		$(if $(LABEL_STT),--label-stt $(LABEL_STT),) \
		$(if $(STT_PRIMARY),--stt-primary,)

# ==============================================================================
# STREAMING STT  —  WebSocket /v1/realtime benchmarks
# ==============================================================================

# Streaming STT sanity check — quick WebSocket smoke test
stt-streaming-sanity:
	$(PYTHON) core/sweep.py --bench stt-streaming-sanity $(if $(LABEL),--label $(LABEL),)

# Streaming STT concurrency benchmark — TTFW, inter-delta, WER under concurrent streams
stt-streaming-bench:
	$(PYTHON) core/sweep.py --bench stt-streaming-bench $(if $(LABEL),--label $(LABEL),)

# ==============================================================================
# BENCH-ONLY  —  run against whatever vLLM server is currently up
# ==============================================================================

bench-sanity:
	$(DC) run --rm bench-runner \
		python /app/core/bench_runner.py --config /configs/sanity_check.yaml

bench-concurrency:
	$(DC) run --rm bench-runner \
		python /app/core/bench_runner.py --config /configs/concurrency_bench.yaml

# ==============================================================================
# UTILITIES
# ==============================================================================

logs:
	$(DC) logs -f vllm-8000

status:
	@echo "=== Containers ==="
	$(DC) ps
	@echo ""
	@echo "=== GPU State ==="
	@nvidia-smi \
		--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
		--format=csv,noheader,nounits

stop:
	$(DC) down

# Build the Gemma 4 vLLM image with Transformers v5.5.3+ (fixes gemma4 architecture)
build-gemma4:
	docker build --no-cache -f Dockerfile.vllm-gemma4 -t vllm/vllm-openai:gemma4-v2 .

# Build the Voxtral-fixed vLLM image with vllm-project/vllm PR #39229 hot-patched.
# This unblocks concurrent /v1/audio/transcriptions requests for Voxtral-Mini-4B.
build-voxtral-fix:
	docker build -f Dockerfile.vllm-voxtral-fixed -t vllm/vllm-openai:voxtral-fix .

results:
	@echo "=== Result Files (most recent first) ==="
	@ls -lht results/ | head -40

gpu-monitor:
	$(DC) --profile monitoring run --rm nvitop

prefetch:
	$(PYTHON) core/prefetch.py

tui:
	$(PYTHON) tui.py

.PHONY: \
	sanity concurrency-bench co-deploy probe serve \
	embed-up embed-down embed-logs embed-sanity \
	stt-sanity stt-bench mixed-co-deploy download-stt-data \
	stt-streaming-sanity stt-streaming-bench \
	bench-sanity bench-concurrency \
	logs status stop build-gemma4 results gpu-monitor prefetch tui
