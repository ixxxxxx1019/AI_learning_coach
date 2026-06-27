"""
  Tutor Agent —— AI 讲师。

  职责：根据 Planner 生成的阶段计划（StudyPhase），
       生成详细的 Markdown 教学内容。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agent.llm import get_llm

TUTOR_SYSTEM = """你是一个专业的AI讲师，擅长用生动易懂的方式讲解知识。

  ## 你的教学风格
  - 用中文讲解，内容清晰有条理
  - 对每个知识点提供：定义、例句/示例、记忆技巧
  - 如果知识点之间有联系，要明确指出
  - 用 Markdown 格式组织内容，包含标题、列表、加粗等
  - 对于词汇类知识，提供：词性、中文释义、英文例句+中文翻译、近义词辨析

  ## 输出格式
  用 Markdown 组织你的教学内容，结构如下：
  ```markdown
  ## 知识点名称1
  ### 定义
  ...
  ### 示例
  ...
  ### 记忆技巧
  ...

  ## 知识点名称2
  ...
  """

TUTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TUTOR_SYSTEM),
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
      test_input = """
      请讲解以下CET6词汇知识点，当前是 learn_new 阶段：

      - k002: sophisticated - 复杂精密的，世故的
      - k004: articulate - 清晰表达的，能说会道的

      这些词汇属于CET6高频词汇，请给出详细的讲解和记忆技巧。
      """

      tutor = create_tutor()
      result = tutor.invoke({"user_input": test_input})
      print(result)



