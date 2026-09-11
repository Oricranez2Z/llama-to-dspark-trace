# Stage 3 — AngelSlim external runner

Status: complete; the subsequent GPU execution is recorded in stages 4 and 5.

## Design

The project does not vendor AngelSlim or declare its CUDA stack as a package
dependency. `experiments/run_dflare_gpu_experiment.py` launches a dedicated
external Python environment and records immutable source provenance.

The low-level integration imports the pinned DFlash/DFlare model modules and
AngelSlim's official `dflash_generate` loop from a source checkout. Namespace
loading avoids importing unrelated training, benchmark-server, Ray, and
DeepSpeed dependencies from AngelSlim's package initializers.

## Portability

- `float16` is the RTX 8000 default.
- `bfloat16` fails early below compute capability 8.0.
- SDPA is the portable default on both RTX 8000 and RTX 4090. The same runner
  accepts `--attention flash_attention_2`; this is the recommended 4090
  performance profile after installing a compatible `flash-attn` build.
- Model paths stay in process arguments. Published JSON stores model labels,
  source commits, and relative runner paths only.

## External environment change

The reused external AngelSlim-compatible environment received `datasets==5.0.1`,
`loguru==0.7.3`, `multiprocess==0.70.19`, `xxhash==4.0.1`, and an update from
`pyarrow==19.0.1` to `pyarrow==25.0.1`. No AngelSlim package was installed.
