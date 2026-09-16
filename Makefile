.PHONY: setup test lint demo demo-spec demo-dflare compare-dflare plot-unified visualize visualize-measured samples clean

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

demo-dflare:
	uv run python experiments/run_dflare_educational.py \
		--output results/generated/dflare_educational.json \
		--trace results/generated/dflare_educational_trace.jsonl

compare-dflare:
	uv run python experiments/compare_dflare_results.py \
		--angelslim-result results/measured/ar_dflash_dflare_rtx8000_fp16_mixed4/result.json \
		--vllm-ar-outputs results/measured/vllm_ar_rtx8000_fp16_control/outputs.json \
		--vllm-dflare-outputs results/measured/vllm_dflare_patch_rtx8000_fp16_smoke/outputs.json \
		--output results/measured/dflare_comparison_rtx8000_fp16.json
	uv run python visualization/plot_dflare_comparison.py \
		results/measured/dflare_comparison_rtx8000_fp16.json \
		results/figures/dflare_rtx8000_fp16_comparison.svg

plot-unified:
	uv run python visualization/plot_unified_spec_comparison.py \
		results/measured/unified_spec_qwen3_8b_rtx8000_fp16/summary.json \
		results/figures/unified_spec_qwen3_8b_rtx8000_fp16.svg

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
