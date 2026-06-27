import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Type, TypeVar

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """读取配置：优先 Streamlit Cloud Secrets，回退到环境变量 (.env)。"""
    # 方式 1: Streamlit Cloud Secrets
    try:
        import streamlit as st
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
        for k in (key, key.lower(), key.upper()):
            val = secrets.get(k, "")
            if val:
                return val
    except Exception:
        pass
    # 方式 2: 本地 .env
    return os.getenv(key, default)

def get_llm(temperature: float = 0.3, model: str = "deepseek-chat") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=_get_secret("DEEPSEEK_API_KEY"),
        base_url=_get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=temperature,
        max_tokens=4096,
    )

T = TypeVar("T", bound=BaseModel)

def get_structured_llm(
      output_model: Type[T],
      temperature: float = 0.1,
      model: str = "deepseek-chat",
) -> ChatOpenAI:
      """获取一个能输出 Pydantic 结构化对象的 LLM。

      这是 LangChain 的杀手级特性：不用手写 JSON 解析，
      LLM 直接返回验证过的 Pydantic 对象。

      使用场景：
      - Planner Agent → output_model=StudyPlan
      - Evaluator Agent → output_model=Quiz / GradingResult / Diagnosis

      Args:
          output_model: Pydantic 模型类，LLM 会按要求输出
          temperature: 结构化输出用低温度 0.1，确保格式精确
          model: 模型名

      Returns:
          配置了 with_structured_output 的 ChatOpenAI 实例
      """
      llm = ChatOpenAI(
          model=model,
          api_key=_get_secret("DEEPSEEK_API_KEY"),
          base_url=_get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
          temperature=temperature,
          max_tokens=4096,
      )
      # method="json_mode" 兼容 DeepSeek（不支持原生 json_schema）
      return llm.with_structured_output(output_model, method="json_mode")
