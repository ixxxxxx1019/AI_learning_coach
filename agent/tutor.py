"""
  Tutor Agent —— AI 讲师。

  职责：根据 Planner 生成的阶段计划（StudyPhase），
       生成详细的 Markdown 教学内容。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.llm import get_llm
from config.logging_config import get_logger
from config.prompts import PromptLoader

logger = get_logger(__name__)
_loader = PromptLoader()

TUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _loader.get_system_prompt("tutor")),
    ("user", "{user_input}"),
])


def create_tutor():
    """创建 Tutor Agent。

    Returns:
        一个可调用的 Runnable，输入 dict，输出 Markdown 字符串。

    Usage:
        tutor = create_tutor()
        markdown_text: str = tutor.invoke({"user_input": "请讲解 k002: sophisticated"})
    """
    llm = get_llm(temperature=0.5)
    return TUTOR_PROMPT | llm | StrOutputParser()

if __name__ == "__main__":
      from config.logging_config import setup_logging
      setup_logging()

      test_input = """
      请讲解以下CET6词汇知识点，当前是 learn_new 阶段：

      - k002: sophisticated - 复杂精密的，世故的
      - k004: articulate - 清晰表达的，能说会道的

      这些词汇属于CET6高频词汇，请给出详细的讲解和记忆技巧。
      """

      tutor = create_tutor()
      result = tutor.invoke({"user_input": test_input})
      logger.info("tutor_test_result", content_preview=result[:200])
      print(result)



