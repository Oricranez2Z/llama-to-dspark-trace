# Contributing

Contributions should preserve the project's educational focus and explicit
separation between simulation and measured production behavior.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Expectations

- Add a focused test for each state transition or invariant.
- Keep CPU demos deterministic and small.
- Do not commit model weights, datasets, or large profiler outputs.
- Record upstream revisions when changing a source-level adapter.
- Label simulated measurements as simulations.
- Include hardware, software, workload, and sampling metadata with benchmarks.
- Preserve third-party notices and citations.

## Pull requests

A pull request should explain:

1. Which execution-level question it answers.
2. What state or data contract changes.
3. Which tests demonstrate correctness.
4. Whether any performance values are measured or simulated.
