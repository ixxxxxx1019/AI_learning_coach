"""
Token 成本追踪 —— 实时计算每次 LLM 调用的费用。

基于 LangChain BaseCallbackHandler，无侵入式追踪：
- 记录每次调用的 input/output token 数
- 按模型定价计算费用（支持 DeepSeek / OpenAI）
- 累积 session 成本

Usage:
    from agent.cost_tracker import CostTracker
    tracker = CostTracker()

    # 绑定到 LLM
    llm = get_llm(callbacks=[tracker])

    # 查看统计
    print(tracker.session_cost)   # 当前 session 总费用
    print(tracker.summary())      # 详细统计摘要
"""

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from config.logging_config import get_logger

logger = get_logger(__name__)

# ============================================================
# 定价表（USD / 1M tokens）
# ============================================================
PRICING = {
    "deepseek-chat":      {"input": 0.14, "output": 0.28},
    "deepseek-reasoner":  {"input": 0.55, "output": 2.19},
    "gpt-4o":             {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":        {"input": 0.15, "output": 0.60},
    # 默认回退价格
    "default":            {"input": 0.14, "output": 0.28},
}


def _get_price(model_name: str) -> dict[str, float]:
    """查找模型定价，支持前缀匹配。"""
    # 精确匹配
    if model_name in PRICING:
        return PRICING[model_name]
    # 前缀匹配（如 deepseek-chat-xxx）
    for prefix, price in PRICING.items():
        if model_name.startswith(prefix):
            return price
    return PRICING["default"]


# ============================================================
# CostTracker
# ============================================================

class CostTracker(BaseCallbackHandler):
    """追踪 LLM 调用的 Token 消耗和费用。

    Usage:
        tracker = CostTracker()
        llm = ChatOpenAI(callbacks=[tracker], ...)
        chain.invoke(...)
        print(f"Cost: ${tracker.session_cost:.4f}")
    """

    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__()
        self.model_name = model_name
        self._pricing = _get_price(model_name)

        # 统计数据
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_cost = 0.0

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """LLM 调用开始时记录。"""
        self.call_count += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 调用结束时计算 Token 和费用。

        从 response.llm_output 中提取 token_usage（OpenAI 兼容格式）。
        """
        token_usage = _extract_token_usage(response)
        if token_usage is None:
            return

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # 计算费用
        input_cost = (input_tokens / 1_000_000) * self._pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self._pricing["output"]
        call_cost = input_cost + output_cost
        self.session_cost += call_cost

        logger.debug(
            "llm_cost",
            model=self.model_name,
            call=self.call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(call_cost, 6),
            cumulative_usd=round(self.session_cost, 6),
        )

    # ---- 统计方法 ----
    def summary(self) -> dict[str, Any]:
        """返回成本统计摘要。"""
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "session_cost_usd": round(self.session_cost, 6),
            "input_price_per_m": self._pricing["input"],
            "output_price_per_m": self._pricing["output"],
        }

    def format_cost(self) -> str:
        """格式化为人类可读的费用字符串。"""
        if self.session_cost < 0.01:
            return f"${self.session_cost:.4f}"
        return f"${self.session_cost:.4f} (~¥{self.session_cost * 7.2:.4f})"

    def reset(self):
        """重置统计（新一轮学习时调用）。"""
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.session_cost = 0.0


def _extract_token_usage(response: LLMResult) -> dict[str, int] | None:
    """从 LLMResult 中提取 token 使用信息。

    兼容多种返回格式：
    - OpenAI 标准: response.llm_output["token_usage"]
    - LangChain 格式: response.llm_output["usage"]
    """
    llm_output = response.llm_output
    if llm_output is None:
        return None

    # 尝试多种可能的 key
    for key in ("token_usage", "usage", "tokens"):
        usage = llm_output.get(key)
        if usage:
            return usage

    # 某些 provider 直接在 llm_output 里
    prompt = llm_output.get("prompt_tokens") or llm_output.get("input_tokens")
    completion = llm_output.get("completion_tokens") or llm_output.get("output_tokens")
    if prompt is not None or completion is not None:
        return {
            "prompt_tokens": prompt or 0,
            "completion_tokens": completion or 0,
        }

    return None
