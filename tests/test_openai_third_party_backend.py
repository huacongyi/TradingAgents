"""OpenAI provider with explicit base_url (e.g. NVIDIA NIM): auth + Responses API."""

from __future__ import annotations

import importlib


def _reload_openai_client():
    import tradingagents.llm_clients.openai_client as mod
    return importlib.reload(mod)


def test_openai_nvidia_base_uses_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    mod = _reload_openai_client()
    client = mod.OpenAIClient(
        model="minimaxai/minimax-m2.7",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="openai",
    )
    llm = client.get_llm()
    assert llm.openai_api_key.get_secret_value() == "sk-test"
    assert llm.use_responses_api is False


def test_openai_nvidia_base_falls_back_to_nvidia_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-nvidia-env")
    mod = _reload_openai_client()
    client = mod.OpenAIClient(
        model="minimaxai/minimax-m2.7",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="openai",
    )
    llm = client.get_llm()
    assert llm.openai_api_key.get_secret_value() == "nvapi-from-nvidia-env"


def test_openai_custom_base_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    mod = _reload_openai_client()
    client = mod.OpenAIClient(
        model="minimaxai/minimax-m2.7",
        base_url="https://integrate.api.nvidia.com/v1",
        provider="openai",
    )
    try:
        client.get_llm()
    except ValueError as e:
        assert "OPENAI_API_KEY" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_openai_default_host_uses_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mod = _reload_openai_client()
    client = mod.OpenAIClient(model="gpt-5.4", provider="openai")
    llm = client.get_llm()
    assert llm.use_responses_api is True
