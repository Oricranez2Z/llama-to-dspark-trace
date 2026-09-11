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

## Checkpoints used by the DFlare track

- Target: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`
- DFlare: `AngelSlim/Qwen3-4b-dflare`, revision
  `71dcbb0645c9357c938ab4e30a677c5a63bff900`, Apache-2.0 model card
- DFlash: `z-lab/Qwen3-4B-DFlash-b16`, revision
  `b74e3a329c4d963783143b1e970d95b002be72bd`; consult its model card for terms

Weights are downloaded into external cache/model directories and are not
committed to this repository.

## Experimental vLLM DFlare patch

- Base: vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- Upstream proposal adapted from:
  <https://github.com/vllm-project/vllm/pull/49023>
- Lab patch: Apache-2.0-compatible changes carried as a formatted patch with
  attribution trailers; the proposal was closed without merge

Model checkpoints, datasets, and generated outputs remain subject to their own
licenses and usage terms. They are not redistributed here.
