"""
Planner Agent —— 学习规划师。

职责：根据用户学科、可用时间、当前进度，
      分析知识图谱中的依赖关系，
      生成结构化的 StudyPlan。
"""
from langchain_core.prompts import ChatPromptTemplate
from agent.llm import get_structured_llm
from agent.models import StudyPlan

PLANNER_SYSTEM = """你是一个专业的AI学习规划师。你的任务是根据学生的学习情况，制定个性化的学习计划。

 ## 你的工作流程
 1. 分析学生提供的知识点列表和依赖关系
 2. 评估哪些知识点应该优先学习（考虑依赖链）
 3. 将学习过程划分为 review（复习）、learn_new（学新知识）、quiz（测验）三个阶段
 4. 为每个阶段分配合理的时长

 ## 规划原则
 - 有前置依赖未掌握的知识点，必须先复习前置知识
 - 每次学习应该包含复习→新学→测验的完整循环
 - 难度高的知识点分配更多时间
 - 每个阶段的时间分配要合理，总时长不超过用户设定的时间

 ## 输出格式
 你必须严格按照 StudyPlan 的 JSON 结构输出，包含：
 - subject_name: 学科名称
 - total_minutes: 总时长
 - rationale: 规划理由
 - phases: 阶段列表，每个阶段包含 type、name、kp_ids、estimated_minutes、instruction
 """

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
      ("system", PLANNER_SYSTEM),
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

      print("=" * 60)
      print(f"学科: {plan.subject_name}")
      print(f"总时长: {plan.total_minutes}分钟")
      print(f"规划理由: {plan.rationale}")
      print(f"阶段数: {len(plan.phases)}")
      for i, phase in enumerate(plan.phases):
          print(f"\n--- 阶段{i+1}: {phase.name} ---")
          print(f"  类型: {phase.type}")
          print(f"  知识点: {phase.kp_ids}")
          print(f"  预计时长: {phase.estimated_minutes}分钟")
          print(f"  指导: {phase.instruction}")