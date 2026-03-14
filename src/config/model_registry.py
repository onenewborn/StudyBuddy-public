"""
Model Registry — 每模块可选模型预设
======================================

支持的模型（在 .env 中配置对应变量）：

  gemini       → Gemini 2.5 Pro   (Google AI，贵但最强，支持视觉)
  gemini-flash → Gemini 2.0 Flash (Google AI，便宜约15x，仍支持视觉，推荐用于图片解题)
  kimi         → Kimi-2.5         (Moonshot AI，需要 temperature=1，支持视觉+文字)
  kimi-coding  → Kimi Coding会员  (Moonshot AI，纯文字/代码，不支持视觉)

.env 配置格式：
  # Gemini 2.5 Pro（复杂推理）
  MODEL_GEMINI_API_KEY=AIza...
  MODEL_GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
  MODEL_GEMINI_MODEL=gemini-2.5-pro-preview-05-06

  # Gemini Flash（图片解题/视觉任务，复用同一 API Key）
  # 不需要单独配置 API Key，自动复用 MODEL_GEMINI_API_KEY 或 LLM_API_KEY
  # MODEL_GEMINI_FLASH_MODEL=gemini-2.0-flash   # 可选，留空则用内置默认

  # Kimi-2.5（temperature 强制=1，否则 API 400，支持视觉+文字）
  MODEL_KIMI_API_KEY=sk-...
  MODEL_KIMI_BASE_URL=https://api.moonshot.cn/v1    # 可留空，使用内置默认
  MODEL_KIMI_MODEL=kimi-k2.5                        # 可留空，使用内置默认

  # Kimi Coding 会员（KIMI_API_KEY，from https://www.kimi.com/code/console）
  # 注意：kimi-coding 使用独立 API 端点，需要特殊 User-Agent: claude-code/1.2.0
  # 仅支持文字/代码，不支持图片
  KIMI_API_KEY=sk-kimi-...
  KIMI_CODING_BASE_URL=https://api.kimi.com/coding/v1   # 可留空，使用内置默认
  KIMI_CODING_MODEL=kimi-for-coding                     # 可留空，使用内置默认
"""

from __future__ import annotations

import os
from typing import Optional

# ── 预设定义 ────────────────────────────────────────────────────────────────
# temperature: None 表示使用 agents.yaml 里各 agent 自身的默认值
#              float 表示该模型要求的固定 temperature（会覆盖 yaml 配置）

SUPPORTED_MODELS: dict[str, dict] = {
    "gemini": {
        "label":            "Gemini 2.5 Pro",
        "api_key_env":      "MODEL_GEMINI_API_KEY",
        "base_url_env":     "MODEL_GEMINI_BASE_URL",
        "model_env":        "MODEL_GEMINI_MODEL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model":    "gemini-2.5-pro-preview-05-06",
        "temperature":      None,       # 使用 agents.yaml 默认
    },
    "gemini-flash": {
        "label":            "Gemini Flash (图片推荐)",
        # Flash 与 Pro 共用同一 API Key：优先读 MODEL_GEMINI_API_KEY，没有则回退全局 LLM_API_KEY
        "api_key_env":      "MODEL_GEMINI_API_KEY",
        "base_url_env":     "MODEL_GEMINI_BASE_URL",
        "model_env":        "MODEL_GEMINI_FLASH_MODEL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model":    "gemini-2.0-flash",   # 约为 Pro 的 1/15 费用，仍支持视觉
        "temperature":      None,
    },
    "kimi": {
        "label":            "Kimi-2.5",
        "api_key_env":      "MODEL_KIMI_API_KEY",
        "base_url_env":     "MODEL_KIMI_BASE_URL",
        "model_env":        "MODEL_KIMI_MODEL",
        "default_base_url": "https://api.moonshot.cn/v1",
        "default_model":    "kimi-k2.5",
        "temperature":      1.0,        # kimi-k2.5 API 强制要求 temperature=1
    },
    "kimi-coding": {
        "label":            "Kimi Coding",
        "api_key_env":      "KIMI_API_KEY",
        "base_url_env":     "KIMI_CODING_BASE_URL",
        "model_env":        "KIMI_CODING_MODEL",
        "default_base_url": "https://api.kimi.com/coding/v1",  # 专属端点，非 moonshot.cn
        "default_model":    "kimi-for-coding",                  # 专属模型名
        "temperature":      None,       # 使用 agents.yaml 默认
        "binding":          "kimi-coding",  # 需要特殊 User-Agent: claude-code/1.2.0
    },
    "kimi-turbo": {
        "label":            "Kimi Turbo（快速文字）",
        "api_key_env":      "MODEL_KIMI_API_KEY",
        "base_url_env":     "MODEL_KIMI_BASE_URL",
        "model_env":        "MODEL_KIMI_TURBO_MODEL",
        "default_base_url": "https://api.moonshot.cn/v1",
        "default_model":    "kimi-k2-turbo-preview",
        "temperature":      1.0,        # moonshot API 强制 temperature=1
    },
}


def get_model_config(model_key: Optional[str]) -> Optional[dict]:
    """
    返回指定预设的完整 LLM 配置字典。

    Returns:
        dict(api_key, base_url, model, label, temperature)，找不到返回 None。
    """
    if not model_key:
        return None

    preset = SUPPORTED_MODELS.get(model_key.strip().lower())
    if not preset:
        return None

    # api_key: 优先使用预设专属 key，不存在时回退到全局 LLM_API_KEY
    api_key  = os.getenv(preset["api_key_env"]) or os.getenv("LLM_API_KEY", "")
    # base_url / model: 绝不回退到 LLM_HOST/LLM_MODEL，避免把 Gemini URL 喂给 Kimi（反之亦然）
    base_url = os.getenv(preset["base_url_env"]) or preset["default_base_url"]
    model    = os.getenv(preset["model_env"])    or preset["default_model"]

    return {
        "api_key":     api_key,
        "base_url":    base_url,
        "model":       model,
        "label":       preset["label"],
        "temperature": preset.get("temperature"),   # None or float
        "binding":     preset.get("binding"),       # None or str (e.g. "kimi-coding")
    }


def agent_kwargs(model_key: Optional[str]) -> dict:
    """
    返回可直接传给 BaseAgent 构造器的 kwargs 字典。

    当 model_key 有效时返回 {api_key, base_url, model, [temperature]}；
    否则返回 {}（Agent 使用默认 LLM 配置）。

    temperature 只在预设有强制要求时才包含在返回字典里，
    由 BaseAgent.__init__ 的 temperature 参数接收并覆盖 agents.yaml。
    """
    cfg = get_model_config(model_key)
    if not cfg:
        return {}
    result: dict = {
        "api_key":  cfg["api_key"],
        "base_url": cfg["base_url"],
        "model":    cfg["model"],
    }
    if cfg["temperature"] is not None:
        result["temperature"] = cfg["temperature"]
    if cfg.get("binding"):
        result["binding"] = cfg["binding"]
    return result


__all__ = ["SUPPORTED_MODELS", "get_model_config", "agent_kwargs"]
