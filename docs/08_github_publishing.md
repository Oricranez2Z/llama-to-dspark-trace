# GitHub publishing checklist

## Before publishing

1. Verify the author metadata in `pyproject.toml`, `LICENSE`, and
   `CITATION.cff`.
2. Run tests and lint.
3. Regenerate the curated examples.
4. Review every benchmark claim and its environment metadata.
5. Confirm that no weights, datasets, tokens, secrets, or absolute local paths
   are tracked.

## Create the GitHub repository

Create an empty repository named `llm-serving-lab`. Do not initialize it with a
README, license, or `.gitignore`, because those files already exist locally.

Then run:

```bash
cd llm-serving-lab
git init
git add .
git status
git commit -m "Initial release: LLM serving execution lab"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/llm-serving-lab.git
git push -u origin main
```

HTTPS alternative:

```bash
git remote add origin https://github.com/YOUR_USERNAME/llm-serving-lab.git
git push -u origin main
```

## After publishing

- Add repository topics: `llm`, `inference`, `vllm`, `kv-cache`,
  `speculative-decoding`, `continuous-batching`.
- Enable GitHub Actions and verify the CI workflow passes.
- Create a `v0.1.0` release from the tested commit.
- Pin the repository on your GitHub profile.
- Add the repository link to the matching resume entry.
