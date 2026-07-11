"""
Prompt 加载器 —— 从 YAML 文件加载系统提示。

将 Prompt 从代码中解耦，实现：
- 非代码人员也能调整 AI 行为
- Prompt 版本可追踪（Git diff 友好）
- 同一 Agent 可支持多套 Prompt（A/B Testing）

Usage:
    from config.prompts import PromptLoader
    loader = PromptLoader()
    system_prompt = loader.get_system_prompt("planner")
"""

import warnings
from functools import lru_cache
from pathlib import Path

import yaml

# ---- 硬编码回退（向后兼容）----
# 这些仅在 YAML 文件缺失时使用，保持与旧版本兼容

_FALLBACK_PLANNER = """你是一个专业的AI学习规划师。你的任务是根据学生的学习情况，制定个性化的学习计划。

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

_FALLBACK_TUTOR = """你是一个专业的AI讲师，擅长用生动易懂的方式讲解知识。

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

_FALLBACK_EVALUATOR = {
    "quiz": """你是一个专业的测评专家。请根据提供的知识点，生成一套测验题。

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
""",
    "grading": """你是一个严格的阅卷老师。请根据标准答案批改学生的作答。

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
  """,
    "diagnosis": """你是一个AI学习诊断专家。根据学生的答题情况，诊断知识点的掌握度变化。

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
  """,
}

# ---- 路径常量 ----
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=8)
def _load_yaml_cached(filepath: str) -> dict | None:
    """模块级缓存函数：加载 YAML 文件（避免 lru_cache 泄漏 self）。"""
    try:
        with Path(filepath).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        warnings.warn(
            f"Failed to load {filepath}: {e}, using hardcoded fallback.",
            stacklevel=2,
        )
        return None


class PromptLoader:
    """从 YAML 文件中加载系统提示。

    特性：
    - 自动缓存已加载的 Prompt（避免重复 I/O）
    - YAML 文件缺失时回退到硬编码默认值
    - 支持 evaluator 的三条子 Chain（quiz / grading / diagnosis）

    Usage:
        loader = PromptLoader()
        prompt = loader.get_system_prompt("planner")         # 单条 Prompt
        prompt = loader.get_system_prompt("evaluator", "quiz")  # 子 Prompt
    """

    def __init__(self, prompts_dir: Path | None = None):
        self._prompts_dir = prompts_dir or _PROMPTS_DIR

    def _load_yaml(self, agent_name: str) -> dict | None:
        """加载 YAML 文件（使用模块级缓存）。"""
        yaml_path = self._prompts_dir / f"{agent_name}.yaml"
        if not yaml_path.exists():
            warnings.warn(
                f"Prompt file not found: {yaml_path}, using hardcoded fallback.",
                stacklevel=3,
            )
            return None
        return _load_yaml_cached(str(yaml_path))

    def get_system_prompt(self, agent_name: str, sub_agent: str | None = None) -> str:
        """获取系统提示文本。

        Args:
            agent_name: Agent 名称（planner / tutor / evaluator）
            sub_agent:  子 Agent（仅 evaluator 需要：quiz / grading / diagnosis）

        Returns:
            System prompt 文本
        """
        data = self._load_yaml(agent_name)

        if data is None:
            # 回退到硬编码默认值
            return self._get_fallback(agent_name, sub_agent)

        if sub_agent:
            # evaluator 的子 Prompt（quiz / grading / diagnosis）
            sub_data = data.get(sub_agent, {})
            if isinstance(sub_data, dict):
                return sub_data.get("system", self._get_fallback(agent_name, sub_agent))
            return str(sub_data)

        return data.get("system", self._get_fallback(agent_name, sub_agent))

    def _get_fallback(self, agent_name: str, sub_agent: str | None = None) -> str:
        """获取硬编码回退 Prompt。"""
        if agent_name == "planner":
            return _FALLBACK_PLANNER
        elif agent_name == "tutor":
            return _FALLBACK_TUTOR
        elif agent_name == "evaluator":
            if sub_agent:
                return _FALLBACK_EVALUATOR.get(sub_agent, "")
            return ""
        return ""

    def clear_cache(self):
        """清除 Prompt 缓存（用于热重载）。"""
        self._load_yaml.cache_clear()
