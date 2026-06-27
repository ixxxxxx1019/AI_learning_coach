import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Type, TypeVar

load_dotenv()

def get_llm(temperature: float = 0.3, model: str = "deepseek-chat") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
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
          api_key=os.getenv("DEEPSEEK_API_KEY"),
          base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
          temperature=temperature,
          max_tokens=4096,
      )
      # method="json_mode" 兼容 DeepSeek（不支持原生 json_schema）
      return llm.with_structured_output(output_model, method="json_mode")
