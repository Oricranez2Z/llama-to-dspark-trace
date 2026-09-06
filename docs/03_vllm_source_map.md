# vLLM source map

This map targets vLLM revision `ff6173997`.

## Offline request path

```text
vllm/entrypoints/llm.py: LLM.generate
→ vllm/v1/engine/llm_engine.py: LLMEngine.add_request
→ vllm/v1/engine/core.py: EngineCore.add_request / step
→ vllm/v1/core/sched/scheduler.py: Scheduler.schedule
→ vllm/v1/core/kv_cache_manager.py: allocate_slots
→ executor / worker / GPUModelRunner
→ sampler
→ Scheduler.update_from_output
→ OutputProcessor.process_outputs
```

## Serving request path

```text
OpenAI API server
→ AsyncLLM.generate
→ EngineCoreClient
→ EngineCore process
→ Scheduler
→ GPU worker
→ EngineCoreOutput
→ OutputProcessor
→ incremental detokenization
→ SSE response
```

## V1 and V2 model runners

The pinned revision contains both runners. Standard tracing can force V1 with:

```bash
VLLM_USE_V2_MODEL_RUNNER=0
```

DSpark forces V2 in the pinned configuration. Read them separately rather than
mixing their data structures in the first pass.

## Local concept map

| Lab component | vLLM concept |
|---|---|
| `Request` | V1 request state |
| `SchedulerOutput` | scheduler-to-worker execution contract |
| `BlockManager` | KV cache manager and block pool |
| `slot_mapping` | physical KV write locations |
| `Engine.step` | schedule, execute, update cycle |
| `TraceRecorder` | request-level execution events |

## Trace adapter

`VLLMTraceAdapter` wraps a small number of Python boundaries. It should be used
only against a known revision and with eager, single-process execution first.
CUDA graphs and multiprocessing should be enabled only after the logical trace
is understood.
