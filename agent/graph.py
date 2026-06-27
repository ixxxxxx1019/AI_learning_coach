from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.models import LearningState
from agent.planner import create_planner
from agent.tutor import create_tutor
from agent.evaluator import create_quiz_chain, create_grading_chain, create_diagnosis_chain


def planner_node(state: LearningState) -> dict:
    """节点1：制定学习计划。

    从 state 中读取学科信息、时间、进度，
    调用 Planner Agent 生成 StudyPlan，
    返回 plan 和 subject_name。
    """
    planner = create_planner()

    # 从状态中拼装 user_input
    user_input = f"""
     学科：{state.get("subject_name", "未指定")}
     学科类型：{state.get("subject_type", "unknown")}
     可用时间：{state.get("time_minutes", 30)}分钟

     """

    # 如果状态中有知识点进度信息，一并传入
    if state.get("progress_info"):
        user_input += f"\n当前学习进度：\n{state['progress_info']}\n"

    plan = planner.invoke({"user_input": user_input})

    print(f"[planner] 计划生成完成: {plan.subject_name}, {len(plan.phases)}个阶段, {plan.total_minutes}分钟")

    return {
        "plan": plan.model_dump(),
        "subject_name": plan.subject_name,
    }

def tutor_node(state: LearningState) -> dict:
      """节点2：AI讲师讲解。

      从 state 中读取 plan，按阶段顺序生成教学内容。
      将所有阶段的讲解结果存入 tutor_results。
      """
      tutor = create_tutor()
      plan = state.get("plan", {})
      phases = plan.get("phases", [])

      results = []
      for i, phase in enumerate(phases):
          user_input = f"""
          当前是第{i+1}阶段：{phase['name']}
          阶段类型：{phase['type']}
          请讲解以下知识点（ID列表）：{', '.join(phase['kp_ids'])}

          指导说明：{phase.get('instruction', '请详细讲解这些知识点')}
          """

          content = tutor.invoke({"user_input": user_input})
          results.append({
              "phase_index": i,
              "phase_name": phase["name"],
              "phase_type": phase["type"],
              "content": content,
          })
          print(f"[tutor] 阶段{i+1}/{len(phases)} 讲解完成: {phase['name']}")

      return {"tutor_results": results}


def quiz_node(state: LearningState) -> dict:
    """节点3：生成测验题。

    根据 tutor_results 中涉及的阶段信息生成 Quiz，
    将出题结果存入 quiz_data。
    """
    quiz_chain = create_quiz_chain()
    plan = state.get("plan", {})
    phases = plan.get("phases", [])

    # 收集所有阶段涉及的知识点
    all_kp_ids = []
    for phase in phases:
        all_kp_ids.extend(phase.get("kp_ids", []))

    user_input = f"""
      请为以下知识点生成测验题：
      {', '.join(all_kp_ids)}

      这些知识点来自 {len(phases)} 个学习阶段，请综合出题，每个知识点至少1道题。
      """

    quiz = quiz_chain.invoke({"user_input": user_input})

    print(f"[quiz] 生成 {len(quiz.questions)} 道题, 预计 {quiz.estimated_total_minutes} 分钟")

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

    graded = grading_chain.invoke({"user_input": user_input})

    print(f"[grade] 批改完成: {graded.total_correct}/{graded.total_questions} 正确, 得分 {graded.overall_score}")

    return {"graded": graded.model_dump()}


def diagnose_node(state: LearningState) -> dict:
    """节点5：学习诊断。

    根据批改结果，调用 Diagnosis Chain 诊断每个知识点的掌握度变化，
    给出下一步学习建议。结果存入 diagnosis。
    """
    diagnosis_chain = create_diagnosis_chain()
    graded = state.get("graded", {})

    user_input = f"""
      批改结果：
      {graded}

      请根据以上批改结果，诊断各知识点的掌握度变化并给出学习建议。
      """

    diag = diagnosis_chain.invoke({"user_input": user_input})

    print(f"[diagnose] 诊断完成: 综合评分 {diag.overall_score}")
    print(f"[diagnose] 下一步优先: {diag.next_priority}")

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
      print("=" * 60)
      print("LangGraph 流程测试")
      print("=" * 60)

      graph = create_graph()

      # 用 thread_id 标识一次会话，MemorySaver 据此持久化状态
      config = {"configurable": {"thread_id": "test-session-001"}}

      # 初始状态：模拟用户输入
      initial_state = {
          "subject_id": "cet6_vocab",
          "subject_name": "CET6英语词汇",
          "subject_type": "vocabulary",
          "time_minutes": 20,
          "progress_info": "已掌握基础词汇（k001: simple），需要学习高频词汇（k002, k003, k004）",
      }

      # 第一次执行：planner → tutor → quiz（到 grade 前暂停）
      print("\n--- 第一阶段：规划 + 讲解 + 出题 ---")
      result = graph.invoke(initial_state, config)

      print("\n[中间状态 - Graph 已在 grade 前暂停]")
      print(f"  计划: {result.get('subject_name')}")
      print(f"  讲解阶段数: {len(result.get('tutor_results', []))}")
      quiz_data = result.get("quiz_data", {})
      print(f"  出题数: {len(quiz_data.get('questions', []))}")

      # 检查 Graph 状态：应该暂停在 grade 之前
      state = graph.get_state(config)
      print(f"  下一个节点: {state.next}")  # 应该是 ('grade',)
      print(f"  已完成的节点: {[n for n in state.metadata.get('writes', {})]}")

      # 模拟用户答题后，继续执行
      print("\n--- 第二阶段：用户答题 → 批改 + 诊断 ---")

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
      print(f"\n[最终结果]")
      print(f"  正确: {graded.get('total_correct')}/{graded.get('total_questions')}")
      print(f"  得分: {graded.get('overall_score')}")
      print(f"  诊断总结: {diagnosis.get('summary', '')[:100]}")
      print(f"  下一步: {diagnosis.get('next_priority')}")
      print(f"  会话完成: {result.get('session_done')}")

      print("\n" + "=" * 60)
      print("流程测试完成！")
