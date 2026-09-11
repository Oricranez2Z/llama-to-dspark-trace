# Third-party references

This repository contains original educational implementations and adapters. It
does not vendor the source code or model weights of the projects below.

## llama2.c

- Project: <https://github.com/karpathy/llama2.c>
- Studied revision: `350e04f`
- License: MIT at the studied revision
- Use: reference for a minimal single-token Llama inference state machine

## vLLM

- Project: <https://github.com/vllm-project/vllm>
- Studied revision: `ff6173997`
- License: Apache-2.0 at the studied revision
- Use: source mapping and optional runtime tracing adapter
- Measured smoke run: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
  with a dirty-state flag preserved in its manifest

## DeepSpec and DSpark

- Project: DeepSpec source checkout
- Studied revision: `005e03b`
- License: MIT at the studied revision, with additional notices in that project
- Use: reference for DSpark model, draft proposal, and verification flow

## AngelSlim and DFlare

- Project: <https://github.com/Tencent/AngelSlim>
- Studied revision: `ee8ddb2b43e20800bcfdda1e9ac34ea2aab5de5d`
- License: Apache-2.0, with third-party notices in that project
- Use: reference for DFlare layer-wise fusion, official checkpoint loading,
  and offline comparison methodology

Model checkpoints, datasets, and generated outputs remain subject to their own
licenses and usage terms. They are not redistributed here.
