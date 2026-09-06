.PHONY: setup test lint demo demo-spec visualize visualize-measured samples clean

setup:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

demo:
	uv run python experiments/run_scheduler_experiment.py \
		--config experiments/configs/scheduler_demo.json \
		--output results/generated/scheduler_trace.jsonl

demo-spec:
	uv run python experiments/run_spec_decode_benchmark.py \
		--config experiments/configs/speculative_demo.json \
		--output results/generated/speculative_summary.json

visualize: demo demo-spec
	uv run python visualization/plot_request_timeline.py \
		results/generated/scheduler_trace.jsonl \
		results/generated/scheduler_timeline.svg
	uv run python visualization/plot_kv_blocks.py \
		results/generated/scheduler_trace.jsonl \
		results/generated/kv_blocks.svg
	uv run python visualization/plot_acceptance.py \
		results/generated/speculative_summary.json \
		results/generated/speculative_acceptance.svg

visualize-measured:
	uv run python visualization/plot_vllm_trace.py \
		results/measured/vllm_gpu_smoke/trace.jsonl \
		results/figures/vllm_scheduler_trace.svg
	uv run python visualization/plot_dspark_trace.py \
		results/measured/dspark_gpu_smoke/trace.rank0.jsonl \
		results/figures/dspark_verification_rounds.svg

samples:
	uv run python experiments/run_scheduler_experiment.py \
		--config experiments/configs/scheduler_demo.json \
		--output results/sample_traces/scheduler_trace.jsonl
	uv run python experiments/run_spec_decode_benchmark.py \
		--config experiments/configs/speculative_demo.json \
		--output results/sample_traces/speculative_summary.json
	uv run python visualization/plot_request_timeline.py \
		results/sample_traces/scheduler_trace.jsonl \
		results/figures/scheduler_timeline.svg
	uv run python visualization/plot_kv_blocks.py \
		results/sample_traces/scheduler_trace.jsonl \
		results/figures/kv_blocks.svg
	uv run python visualization/plot_acceptance.py \
		results/sample_traces/speculative_summary.json \
		results/figures/speculative_acceptance.svg

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage results/generated
