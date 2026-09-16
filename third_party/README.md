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

## Unified speculative comparison

- vLLM base: `ff6173997d54c5027971df8ecd1280f046a832b3`
- patched revision: `7160b69e68eca230d26e6e72dbc72c3886741799`
- target: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`
- EAGLE3: `RedHatAI/Qwen3-8B-speculator.eagle3@08610ffa01dd9f16731fe8f627b85905b6aa51c4`
- DFlash: `z-lab/Qwen3-8B-DFlash-b16@9b41424b7109f9c5413454f481b09a82b85333f4`
- DFlare: `AngelSlim/Qwen3-8b-dflare@55e2c8d86d76ce1e79fa3b8642c7f80091285a82`
- DSpark: a local conversion of the DeepSpec Qwen3-8B block-7 checkpoint;
  weights are not redistributed

The formatted Apache-2.0-compatible source patch series is stored in
`integrations/vllm/patches/unified/`.
