from pathlib import Path

PATCH = (
    Path(__file__).parents[1]
    / "integrations"
    / "vllm"
    / "patches"
    / "0001-Add-experimental-Qwen3-DFlare-V1-support.patch"
)


def test_vllm_patch_contains_required_integration_boundaries() -> None:
    content = PATCH.read_text(encoding="utf-8")

    assert "qwen3_dflare.py" in content
    assert 'return self.method == "dflare"' in content
    assert "DFlashProposer" in content
    assert "QwenDFlareDraftModel" in content
    assert "Adapted-from: https://github.com/vllm-project/vllm/pull/49023" in content
