from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.evaluator import create_diagnosis_chain, create_grading_chain, create_quiz_chain
from agent.models import LearningState
from agent.planner import create_planner
from agent.resilience import CircuitBreaker, retryable_invoke
from agent.tutor import create_tutor
from config.logging_config import get_logger
from utils.knowledge_graph import get_kp, list_all_kps, load_kg

logger = get_logger(__name__)

# 为每个 LLM 节点创建独立的熔断器实例
_planner_cb = CircuitBreaker(name="planner", failure_threshold=3, recovery_timeout=30)
_tutor_cb = CircuitBreaker(name="tutor", failure_threshold=3, recovery_timeout=30)
_quiz_cb = CircuitBreaker(name="quiz", failure_threshold=3, recovery_timeout=30)
_grade_cb = CircuitBreaker(name="grade", failure_threshold=3, recovery_timeout=30)
_diagnose_cb = CircuitBreaker(name="diagnose", failure_threshold=3, recovery_timeout=30)


def planner_node(state: LearningState) -> dict:
    """节点1：制定学习计划。

    从 state 中读取学科信息，从知识图谱加载实际知识点，
    调用 Planner Agent 生成 StudyPlan。
    """
    planner = create_planner()
    subject_id = state.get("subject_id", "")
    subject_name = state.get("subject_name", "未指定")

    # 从知识图谱加载该学科的知识点列表
    kg = load_kg()
    all_kps = list_all_kps(kg, subject_id=subject_id)

    # 构建知识点摘要（id + title + difficulty + 依赖关系）
    kp_summary_lines = []
    for kp in all_kps:
        prereqs = kp.get("prerequisites", [])
        prereq_str = f" (前置依赖: {', '.join(prereqs)})" if prereqs else ""
        difficulty_stars = "★" * kp.get("difficulty", 3)
        kp_summary_lines.append(
            f"  - {kp['id']}: {kp['title']} [{difficulty_stars}]{prereq_str}"
        )
    kp_summary = "\n".join(kp_summary_lines) if kp_summary_lines else "（无知识点数据）"

    # 拼装 user_input
    user_input = f"""
学科：{subject_name}
学科类型：{state.get("subject_type", "unknown")}
可用时间：{state.get("time_minutes", 30)}分钟

可用知识点列表（共 {len(all_kps)} 个）：
{kp_summary}

注意：你规划的每个 phase 中的 kp_ids 必须从上述列表中选择，不要编造不存在的 ID。
"""

    # 如果状态中有知识点进度信息，一并传入
    if state.get("progress_info"):
        user_input += f"\n当前学习进度：\n{state['progress_info']}\n"

    plan = retryable_invoke(planner, {"user_input": user_input})

    logger.info(
        "planner_done",
        subject=plan.subject_name,
        phases=len(plan.phases),
        total_minutes=plan.total_minutes,
    )

    return {
        "plan": plan.model_dump(),
        "subject_name": plan.subject_name,
    }

def tutor_node(state: LearningState) -> dict:
      """节点2：AI讲师讲解。

      从 state 中读取 plan，按阶段顺序生成教学内容。
      从知识图谱查找每个 KP 的详细信息（title、定义等）传给 Tutor。
      """
      tutor = create_tutor()
      plan = state.get("plan", {})
      phases = plan.get("phases", [])
      kg = load_kg()
      subject_type = state.get("subject_type", "unknown")

      results = []
      for i, phase in enumerate(phases):
          # 从知识图谱查询每个 KP 的详细信息
          kp_details_lines = []
          for kp_id in phase.get("kp_ids", []):
              kp = get_kp(kg, kp_id)
              if kp:
                  content = kp.get("content", {})
                  title = kp.get("title", kp_id)
                  if subject_type == "vocabulary":
                      word = content.get("word", title)
                      definition = content.get("definition_cn", "")
                      pos = content.get("pos", "")
                      kp_details_lines.append(
                          f"  - {kp_id}: {word} ({pos}) — {definition}"
                      )
                  else:
                      definition = content.get("definition", "")
                      kp_details_lines.append(
                          f"  - {kp_id}: {title} — {definition[:80]}"
                      )
              else:
                  kp_details_lines.append(f"  - {kp_id}: （请根据 ID 推断内容进行讲解）")

          kp_details = "\n".join(kp_details_lines) if kp_details_lines else "（无知识点数据）"

          user_input = f"""
当前是第{i+1}阶段：{phase['name']}
阶段类型：{phase['type']}
学科类型：{subject_type}

请讲解以下知识点：
{kp_details}

指导说明：{phase.get('instruction', '请详细讲解这些知识点')}
"""

          content = retryable_invoke(tutor, {"user_input": user_input})
          results.append({
              "phase_index": i,
              "phase_name": phase["name"],
              "phase_type": phase["type"],
              "content": content,
          })
          logger.debug(
              "tutor_phase_done",
              phase=i + 1,
              total=len(phases),
              phase_name=phase["name"],
          )

      return {"tutor_results": results}


def quiz_node(state: LearningState) -> dict:
    """节点3：生成测验题。

    根据 plan 中涉及的知识点生成 Quiz，
    从知识图谱查询 KP 详情以确保题目准确。
    """
    quiz_chain = create_quiz_chain()
    plan = state.get("plan", {})
    phases = plan.get("phases", [])
    kg = load_kg()
    subject_type = state.get("subject_type", "unknown")

    # 收集所有阶段涉及的知识点 + 查询详情
    all_kp_ids = []
    kp_details_lines = []
    seen = set()
    for phase in phases:
        for kp_id in phase.get("kp_ids", []):
            if kp_id not in seen:
                seen.add(kp_id)
                all_kp_ids.append(kp_id)
                kp = get_kp(kg, kp_id)
                if kp:
                    content = kp.get("content", {})
                    title = kp.get("title", kp_id)
                    if subject_type == "vocabulary":
                        word = content.get("word", title)
                        definition = content.get("definition_cn", "")
                        kp_details_lines.append(f"  - {kp_id}: {word} — {definition}")
                    else:
                        definition = content.get("definition", "")
                        kp_details_lines.append(f"  - {kp_id}: {title} — {definition[:80]}")
                else:
                    kp_details_lines.append(f"  - {kp_id}")

    kp_details = "\n".join(kp_details_lines) if kp_details_lines else "（无知识点数据）"

    user_input = f"""
请为以下知识点生成测验题：

{kp_details}

这些知识点来自 {len(phases)} 个学习阶段（{subject_type}），请综合出题，每个知识点至少1道题。
题目的 target_kp_id 字段必须使用上述列表中的 ID。
"""

    quiz = retryable_invoke(quiz_chain, {"user_input": user_input})

    logger.info("quiz_generated", question_count=len(quiz.questions), estimated_minutes=quiz.estimated_total_minutes)

    return {"quiz_data": quiz.model_dump()}


def grade_node(state: LearningState) -> dict:
    """节点4：批改答案。

    从 state 中读取 quiz_data 和 user_answers（用户提交的答案），
    调用 Grading Chain 批改，结果存入 graded。
    """
    grading_chain = create_grading_chain()
    quiz_data = state.get("quiz_data", {})
    user_answers = state.get("user_answers", {})

    # 拼装批改输入：题目 + 标准答案 + 用户答案
    questions = quiz_data.get("questions", [])
    answer_text = ""
    for q in questions:
        qid = q["id"]
        user_ans = user_answers.get(qid, "未作答")
        correct_ans = q.get("correct", "")
        answer_text += f"- {qid} ({q['type']}): {q['stem']}\n"
        answer_text += f"  用户答案: {user_ans}\n"
        answer_text += f"  正确答案: {correct_ans}\n\n"

    user_input = f"""
      批改以下答题：

      题目与用户答案：
      {answer_text}
      """

    graded = retryable_invoke(grading_chain, {"user_input": user_input})

    logger.info(
        "grading_done",
        correct=graded.total_correct,
        total=graded.total_questions,
        score=graded.overall_score,
    )

    return {"graded": graded.model_dump()}


def diagnose_node(state: LearningState) -> dict:
    """节点5：学习诊断。

    根据批改结果和实际考察的知识点，调用 Diagnosis Chain。
    收集 quiz 中的 target_kp_id 作为有效 ID 集合传入。
    """
    diagnosis_chain = create_diagnosis_chain()
    graded = state.get("graded", {})
    quiz_data = state.get("quiz_data", {})

    # 收集实际考察的知识点 ID + 标题
    valid_kp_ids = set()
    kp_info_lines = []
    for q in quiz_data.get("questions", []):
        kp_id = q.get("target_kp_id", "")
        kp_title = q.get("target_kp_title", "")
        if kp_id and kp_id not in valid_kp_ids:
            valid_kp_ids.add(kp_id)
            kp_info_lines.append(f"  - {kp_id}: {kp_title}")

    kp_id_list = "\n".join(kp_info_lines) if kp_info_lines else "（无）"

    user_input = f"""
批改结果：
{graded}

本次测验考察的知识点（kp_diagnosis 中的 kp_id 必须从以下列表中选择）：
{kp_id_list}

请根据以上批改结果，诊断各知识点的掌握度变化并给出学习建议。
next_priority 中的 ID 也必须从上述考察知识点中选择。
"""

    diag = retryable_invoke(diagnosis_chain, {"user_input": user_input})

    logger.info("diagnosis_done", overall_score=diag.overall_score, next_priority=diag.next_priority)

    return {
        "diagnosis": diag.model_dump(),
        "session_done": True,
    }

def create_graph():
      """构建 AI学习教练 的 LangGraph 流程。

      节点流转:
          planner → tutor → quiz → [等待用户答题] → grade → diagnose → END

      关键设计：
          interrupt_before=["grade"] 让流程在出题后暂停，
          等用户在 Streamlit 界面提交答案后，再继续批改+诊断。
      """
      # 1. 创建 StateGraph，绑定 LearningState 作为共享状态
      workflow = StateGraph(LearningState)
      # 2. 注册所有节点
      workflow.add_node("planner", planner_node)
      workflow.add_node("tutor", tutor_node)
      workflow.add_node("quiz", quiz_node)
      workflow.add_node("grade", grade_node)
      workflow.add_node("diagnose", diagnose_node)
      # 3. 定义节点间的边（流转规则）
      workflow.set_entry_point("planner")  # 入口：planner
      workflow.add_edge("planner", "tutor")  # planner → tutor
      workflow.add_edge("tutor", "quiz")  # tutor → quiz
      workflow.add_edge("quiz", "grade")  # quiz → grade（中间会暂停）
      workflow.add_edge("grade", "diagnose")  # grade → diagnose
      workflow.add_edge("diagnose", END)  # diagnose → 结束
      # 4. 在 grade 节点前暂停，等待用户提交答案
      # MemorySaver 让 Graph 能记住暂停位置，支持中断后恢复
      return workflow.compile(
          interrupt_before=["grade"],
          checkpointer=MemorySaver(),
      )


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
      from config.logging_config import setup_logging
      setup_logging()
      test_logger = get_logger("graph_test")

      test_logger.info("langgraph_test_start")

      graph = create_graph()

      # 用 thread_id 标识一次会话，MemorySaver 据此持久化状态
      config = {"configurable": {"thread_id": "test-session-001"}}

      # 初始状态：模拟用户输入（不再需要硬编码 progress_info，KG 自动提供）
      initial_state = {
          "subject_id": "cet6_vocab",
          "subject_name": "CET6英语词汇",
          "subject_type": "vocabulary",
          "time_minutes": 20,
      }

      # 第一次执行：planner → tutor → quiz（到 grade 前暂停）
      test_logger.info("phase1_start", phase="planning+tutoring+quiz")
      result = graph.invoke(initial_state, config)

      quiz_data = result.get("quiz_data", {})
      test_logger.info(
          "phase1_intermediate",
          subject=result.get("subject_name"),
          tutor_phases=len(result.get("tutor_results", [])),
          quiz_count=len(quiz_data.get("questions", [])),
      )

      # 检查 Graph 状态：应该暂停在 grade 之前
      state = graph.get_state(config)
      test_logger.info(
          "graph_state_check",
          next_node=str(state.next),
          completed_nodes=list(state.metadata.get("writes", {})),
      )

      # 模拟用户答题后，继续执行
      test_logger.info("phase2_start", phase="grading+diagnosis")

      # 模拟用户答案（故意错一半，验证批改能力）
      questions = quiz_data.get("questions", [])
      user_answers = {}
      for i, q in enumerate(questions):
          if i % 2 == 0:
              user_answers[q["id"]] = q.get("correct", "")  # 偶数题答对
          else:
              user_answers[q["id"]] = "我不知道"  # 奇数题故意答错

      # 1) 先更新状态，注入用户答案
      graph.update_state(config, {"user_answers": user_answers})
      # 2) 再用 invoke(None, config) 从暂停点恢复执行
      result = graph.invoke(None, config)

      graded = result.get("graded", {})
      diagnosis = result.get("diagnosis", {})
      test_logger.info(
          "phase2_result",
          correct=f"{graded.get('total_correct')}/{graded.get('total_questions')}",
          score=graded.get("overall_score"),
          diagnosis_summary=diagnosis.get("summary", "")[:100],
          next_priority=diagnosis.get("next_priority"),
          session_done=result.get("session_done"),
      )

      test_logger.info("langgraph_test_done")
