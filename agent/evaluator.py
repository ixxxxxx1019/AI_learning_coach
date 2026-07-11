"""
 Evaluator Agent —— 测评师。

 三条 Chain，各司其职：
 1. 出题 —— 根据知识点生成 Quiz
 2. 批改 —— 对用户答案评分 + 分析
 3. 诊断 —— 根据答题情况更新掌握度 + 给出建议
 """

from langchain_core.prompts import ChatPromptTemplate

from agent.llm import get_structured_llm
from agent.models import Diagnosis, GradingResult, Quiz
from config.logging_config import get_logger
from config.prompts import PromptLoader

logger = get_logger(__name__)
_loader = PromptLoader()

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
      ("system", _loader.get_system_prompt("evaluator", "quiz")),
      ("user", "{user_input}"),
  ])


def create_quiz_chain():
    """创建出题 Chain，输出 Quiz 对象。"""
    return QUIZ_PROMPT | get_structured_llm(Quiz)

GRADING_PROMPT = ChatPromptTemplate.from_messages([
      ("system", _loader.get_system_prompt("evaluator", "grading")),
      ("user", "{user_input}"),
  ])


def create_grading_chain():
    """创建批改 Chain，输出 GradingResult 对象。"""
    return GRADING_PROMPT | get_structured_llm(GradingResult)

DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages([
      ("system", _loader.get_system_prompt("evaluator", "diagnosis")),
      ("user", "{user_input}"),
  ])


def create_diagnosis_chain():
    """创建诊断 Chain，输出 Diagnosis 对象。"""
    return DIAGNOSIS_PROMPT | get_structured_llm(Diagnosis)

if __name__ == "__main__":
      from config.logging_config import setup_logging
      setup_logging()

      # ===== 测试1：出题 =====
      logger.info("test_quiz_start")

      quiz_chain = create_quiz_chain()
      quiz = quiz_chain.invoke({
          "user_input": """
          请为以下CET6词汇知识点出题：
          - k002: sophisticated - 复杂精密的，世故的（难度：hard）
          - k004: articulate - 清晰表达的，能说会道的（难度：medium）
          每个知识点出2道题，题型包含选择题和翻译题。
          """
      })
      logger.info("quiz_generated", count=len(quiz.questions), estimated_minutes=quiz.estimated_total_minutes)
      for q in quiz.questions:
          logger.debug("question", id=q.id, type=q.type, stem_preview=q.stem[:50], answer=q.correct)

      # ===== 测试2：批改 =====
      logger.info("test_grading_start")

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
      logger.info("grading_done", score=graded.overall_score, correct=f"{graded.total_correct}/{graded.total_questions}")
      for gq in graded.graded_questions:
          status = "PASS" if gq.is_correct else "FAIL"
          logger.debug("graded_question", id=gq.question_id, status=status, feedback=gq.feedback[:60])

      # ===== 测试3：诊断 =====
      logger.info("test_diagnosis_start")

      diagnosis_chain = create_diagnosis_chain()
      diag = diagnosis_chain.invoke({
          "user_input": f"""
            批改结果：
            {graded.model_dump_json(indent=2)}

            请根据以上批改结果，诊断各知识点的掌握度变化并给出学习建议。
            """
      })
      logger.info(
          "diagnosis_done",
          overall_score=diag.overall_score,
          next_priority=diag.next_priority,
          summary=diag.summary,
      )
      for kp in diag.kp_diagnosis:
          logger.info(
              "kp_diagnosis",
              kp_id=kp.kp_id,
              title=kp.kp_title,
              mastery_change=kp.mastery_change,
              error_type=kp.error_type,
          )
