"""
LLM 工厂模块。

提供统一的 LLM 实例化入口，支持：
- 普通文本输出的 ChatOpenAI
- 结构化输出（Pydantic 模型）的 ChatOpenAI
- 三种密钥来源：Streamlit Secrets → 环境变量 → .env 文件

Usage:
    from agent.llm import get_llm, get_structured_llm

    llm = get_llm(temperature=0.5)
    structured_llm = get_structured_llm(StudyPlan)
"""

import os
import warnings
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config.settings import get_settings

# ---- 初始化配置 ----
settings = get_settings()

# LangSmith 集成（可选）
if settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint


# ---- 向后兼容：旧版 _get_secret() ----
def _get_secret(key: str, default: str = "") -> str:
    """[已弃用] 读取配置：优先 Streamlit Cloud Secrets，回退到环境变量。

    请改用 config.settings.get_settings() 获取配置。
    """
    warnings.warn(
        "_get_secret() is deprecated, use config.settings.get_settings() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 方式 1: Streamlit Cloud Secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            secrets = dict(st.secrets)
            for k in (key, key.lower(), key.upper()):
                val = secrets.get(k, "")
                if val:
                    return val
    except Exception:
        pass
    # 方式 2: 环境变量 / .env
    return os.getenv(key, default)


# ---- LLM 工厂 ----
T = TypeVar("T", bound=BaseModel)


def get_llm(
    temperature: float | None = None,
    model: str | None = None,
    callbacks: list | None = None,
) -> ChatOpenAI:
    """获取标准 ChatOpenAI 实例（文本输出）。

    Args:
        temperature: 覆盖默认 temperature（None 则使用 Settings 中的 creative 值）
        model:       覆盖默认模型
        callbacks:   LangChain callbacks（用于追踪、成本计算等）

    Returns:
        配置好的 ChatOpenAI 实例
    """
    if temperature is None:
        temperature = settings.llm_temperature_creative
    if model is None:
        model = settings.default_model

    kwargs = {
        "model": model,
        "api_key": settings.deepseek_api_key.get_secret_value(),
        "base_url": settings.deepseek_base_url,
        "temperature": temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    if callbacks:
        kwargs["callbacks"] = callbacks

    return ChatOpenAI(**kwargs)


def get_structured_llm[T: BaseModel](
    output_model: type[T],
    temperature: float | None = None,
    model: str | None = None,
    callbacks: list | None = None,
) -> ChatOpenAI:
    """获取结构化输出的 ChatOpenAI 实例。

    这是 LangChain 的核心特性：LLM 直接返回验证过的 Pydantic 对象，
    无需手写 JSON 解析。

    使用场景：
        - Planner Agent    → output_model=StudyPlan
        - Quiz Chain       → output_model=Quiz
        - Grading Chain    → output_model=GradingResult
        - Diagnosis Chain  → output_model=Diagnosis

    Args:
        output_model: Pydantic 模型类
        temperature:  结构化输出建议用低温度 0.1（确保格式精确）
        model:        模型名
        callbacks:    LangChain callbacks

    Returns:
        配置了 with_structured_output 的 ChatOpenAI 实例
    """
    if temperature is None:
        temperature = settings.llm_temperature_structured
    if model is None:
        model = settings.default_model

    kwargs = {
        "model": model,
        "api_key": settings.deepseek_api_key.get_secret_value(),
        "base_url": settings.deepseek_base_url,
        "temperature": temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    if callbacks:
        kwargs["callbacks"] = callbacks

    llm = ChatOpenAI(**kwargs)
    # method="json_mode" 兼容 DeepSeek（不支持原生 json_schema）
    return llm.with_structured_output(output_model, method="json_mode")
