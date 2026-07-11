"""
Planner Agent —— 学习规划师。

职责：根据用户学科、可用时间、当前进度，
      分析知识图谱中的依赖关系，
      生成结构化的 StudyPlan。
"""
from langchain_core.prompts import ChatPromptTemplate

from agent.llm import get_structured_llm
from agent.models import StudyPlan
from config.logging_config import get_logger
from config.prompts import PromptLoader

logger = get_logger(__name__)
_loader = PromptLoader()

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
      ("system", _loader.get_system_prompt("planner")),
      ("user", "{user_input}"),
  ])


def create_planner():
    """创建 Planner Agent。

    Returns:
        一个可调用的 Runnable，输入 dict，输出 StudyPlan 对象。

    Usage:
        planner = create_planner()
        result: StudyPlan = planner.invoke({"user_input": "..."})
    """
    structured_llm = get_structured_llm(StudyPlan)
    # 用 | 管道符把 Prompt 和 LLM 串成一条 Chain
    return PLANNER_PROMPT | structured_llm

if __name__ == "__main__":
      from config.logging_config import setup_logging
      setup_logging()

      # 模拟用户输入
      test_input = """
      学科：CET6英语词汇
      可用时间：30分钟
      当前进度：已掌握基础词汇（k001），需要学习高频词汇（k002, k003, k004）

      知识点依赖关系：
      - k002 (sophisticated) 依赖 k001 (simple)
      - k003 (eloquent) 依赖 k002 (sophisticated)
      - k004 (articulate) 无依赖

      知识点详情：
      - k001: simple - 简单的（已掌握，mastery=0.9）
      - k002: sophisticated - 复杂精密的（未学）
      - k003: eloquent - 雄辩的（未学）
      - k004: articulate - 清晰表达的（未学）
      """

      planner = create_planner()
      plan = planner.invoke({"user_input": test_input})

      logger.info("planner_test_result")
      logger.info("subject", name=plan.subject_name)
      logger.info("total_minutes", minutes=plan.total_minutes)
      logger.info("rationale", text=plan.rationale)
      logger.info("phase_count", count=len(plan.phases))
      for i, phase in enumerate(plan.phases):
          logger.info(
              f"phase_{i+1}",
              name=phase.name,
              type=phase.type,
              kp_ids=phase.kp_ids,
              estimated_minutes=phase.estimated_minutes,
              instruction=phase.instruction,
          )
