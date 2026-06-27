"""
 Evaluator Agent —— 测评师。

 三条 Chain，各司其职：
 1. 出题 —— 根据知识点生成 Quiz
 2. 批改 —— 对用户答案评分 + 分析
 3. 诊断 —— 根据答题情况更新掌握度 + 给出建议
 """

from langchain_core.prompts import ChatPromptTemplate
from agent.llm import get_structured_llm
from agent.models import Quiz, GradingResult, Diagnosis

QUIZ_SYSTEM = """你是一个专业的测评专家。请根据提供的知识点，生成一套测验题。

## 输出JSON格式要求
你必须输出以下结构的JSON，字段名必须严格一致：
{{
  "questions": [
    {{
      "id": "q1",
      "type": "multiple_choice",
      "stem": "题目主干内容",
      "options": ["A选项", "B选项", "C选项", "D选项"],
      "correct": "正确答案",
      "explanation": "答案解析",
      "target_kp_id": "考察的知识点ID",
      "target_kp_title": "考察的知识点名",
      "difficulty": "medium"
    }}
  ],
  "estimated_total_minutes": 15
}}

## 出题要求
- 每个知识点至少出1道题
- 题型多样化：选择题(multiple_choice)、填空题(fill_blank)、翻译题(translation)
- 难度适中，能检测出学生是否真正掌握
- 每道题都要有清晰的答案解析
- 题目ID格式：q1, q2, q3...
- 请以JSON格式输出
"""

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
      ("system", QUIZ_SYSTEM),
      ("user", "{user_input}"),
  ])


def create_quiz_chain():
    """创建出题 Chain，输出 Quiz 对象。"""
    return QUIZ_PROMPT | get_structured_llm(Quiz)

GRADING_SYSTEM = """你是一个严格的阅卷老师。请根据标准答案批改学生的作答。

## 输出JSON格式要求
你必须输出以下结构的JSON，字段名必须严格一致：
{{
  "graded_questions": [
    {{
      "question_id": "q1",
      "user_answer": "学生提交的答案",
      "correct_answer": "正确答案",
      "is_correct": true,
      "error_analysis": "错误分析",
      "feedback": "评语",
      "target_kp_id": "考察的知识点ID"
    }}
  ],
  "overall_score": 75.0,
  "total_correct": 3,
  "total_questions": 4
}}

## 批改要求
- 对照正确答案，判断每道题是否正确
- 对错题进行错误分析：为什么错、哪里没掌握
- 对每道题给出反馈：鼓励性评语或改进建议
- 计算总分（百分制）
- 请以JSON格式输出
  """

GRADING_PROMPT = ChatPromptTemplate.from_messages([
      ("system", GRADING_SYSTEM),
      ("user", "{user_input}"),
  ])


def create_grading_chain():
    """创建批改 Chain，输出 GradingResult 对象。"""
    return GRADING_PROMPT | get_structured_llm(GradingResult)

DIAGNOSIS_SYSTEM = """你是一个AI学习诊断专家。根据学生的答题情况，诊断知识点的掌握度变化。

## 输出JSON格式要求
你必须输出以下结构的JSON，字段名必须严格一致：
{{
  "overall_score": 75.0,
  "kp_diagnosis": [
    {{
      "kp_id": "k002",
      "kp_title": "知识点名称",
      "mastery_change": 0.1,
      "error_type": "概念不清",
      "detail": "详细分析",
      "recommendation": "学习建议"
    }}
  ],
  "next_priority": ["k002", "k003"],
  "summary": "整体学习诊断总结"
}}

## 诊断要求
- 对每个考察的知识点，估计掌握度变化（mastery_change，范围 -0.3 ~ +0.3）
  - 全部答对 → +0.1 ~ +0.2
  - 全部答错 → -0.2 ~ -0.3
  - 部分正确 → +0.05 ~ +0.1
- 分析错误类型：概念不清 / 拼写错误 / 语法错误 / 词汇混淆 等
- 给出针对性的学习建议
- 指出下一步应优先复习的知识点
- 请以JSON格式输出
  """

DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages([
      ("system", DIAGNOSIS_SYSTEM),
      ("user", "{user_input}"),
  ])


def create_diagnosis_chain():
    """创建诊断 Chain，输出 Diagnosis 对象。"""
    return DIAGNOSIS_PROMPT | get_structured_llm(Diagnosis)

if __name__ == "__main__":
      # ===== 测试1：出题 =====
      print("=" * 60)
      print("测试1：生成测验题")
      print("=" * 60)

      quiz_chain = create_quiz_chain()
      quiz = quiz_chain.invoke({
          "user_input": """
          请为以下CET6词汇知识点出题：
          - k002: sophisticated - 复杂精密的，世故的（难度：hard）
          - k004: articulate - 清晰表达的，能说会道的（难度：medium）
          每个知识点出2道题，题型包含选择题和翻译题。
          """
      })
      print(f"生成 {len(quiz.questions)} 道题，预计 {quiz.estimated_total_minutes} 分钟")
      for q in quiz.questions:
          print(f"  {q.id}: [{q.type}] {q.stem[:50]}... (答案: {q.correct})")

      # ===== 测试2：批改 =====
      print("\n" + "=" * 60)
      print("测试2：批改答案")
      print("=" * 60)

      grading_chain = create_grading_chain()
      # 模拟用户答题
      user_answers = """
        题目与用户答案：
        - q1 (选择题): sophisticated的近义词是？ 用户答案：B. complex  正确答案：B. complex
        - q2 (翻译题): 他是一位见过世面的人  用户答案：He is a sophisticated man  正确答案：He is a sophisticated man
        - q3 (选择题): articulate的含义是？ 用户答案：A. 复杂的  正确答案：C. 清晰表达的
        - q4 (翻译题): 她清晰表达了自己的观点  用户答案：She speak her idea clearly  正确答案：She articulated her opinion clearly
        """
      graded = grading_chain.invoke({"user_input": user_answers})
      print(f"总分: {graded.overall_score}, 正确: {graded.total_correct}/{graded.total_questions}")
      for gq in graded.graded_questions:
          status = "[PASS]" if gq.is_correct else "[FAIL]"
          print(f"  {gq.question_id} {status}: {gq.feedback[:60]}")

      # ===== 测试3：诊断 =====
      print("\n" + "=" * 60)
      print("测试3：学习诊断")
      print("=" * 60)

      diagnosis_chain = create_diagnosis_chain()
      diag = diagnosis_chain.invoke({
          "user_input": f"""
            批改结果：
            {graded.model_dump_json(indent=2)}

            请根据以上批改结果，诊断各知识点的掌握度变化并给出学习建议。
            """
      })
      print(f"综合评分: {diag.overall_score}")
      print(f"下一步优先级: {diag.next_priority}")
      for kp in diag.kp_diagnosis:
          print(f"  {kp.kp_id} ({kp.kp_title}): mastery_change={kp.mastery_change}, error={kp.error_type}")
      print(f"总结: {diag.summary}")