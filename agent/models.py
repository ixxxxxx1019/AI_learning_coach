from pydantic import BaseModel,Field
from typing import TypedDict,Optional

class StudyPhase(BaseModel):
    """学习计划的一个阶段"""
    type: str = Field(description="阶段类型: review / learn_new / quiz")
    name: str = Field(description="阶段名称")
    kp_ids: list[str] = Field(description="该阶段要学习的知识点ID列表")
    estimated_minutes: int = Field(description="预估耗时(分钟)")
    instruction: str = Field(description="该阶段的指导说明")


class StudyPlan(BaseModel):
    subject_name: str = Field(description="学科名称")
    total_minutes: int = Field(description="总时长(分钟)")
    rationale: str = Field(description="为什么这样安排")
    phases: list[StudyPhase] = Field(description="学习阶段列表")

class Question(BaseModel):
    id: str = Field(description="题目ID，如 q1")
    type: str = Field(description="题型: multiple_choice / fill_blank / translation")
    stem: str = Field(description="题目主干")
    options: list[str] = Field(default_factory=list, description="选项列表（选择题用）")
    correct: str = Field(description="正确答案")
    explanation: str = Field(description="答案解析")
    target_kp_id: str = Field(description="考察的知识点ID")
    target_kp_title: str = Field(description="考察的知识点名")
    difficulty: str = Field(default="medium", description="难度: easy / medium / hard")

class Quiz(BaseModel):
    questions: list[Question] = Field(description="题目列表")
    estimated_total_minutes: int = Field(description="预估总耗时(分钟)")

class GradedQuestion(BaseModel):
      question_id: str
      user_answer: str
      correct_answer: str
      is_correct: bool
      error_analysis: str = ""
      feedback: str = ""
      target_kp_id: str = ""

class GradingResult(BaseModel):
      graded_questions: list[GradedQuestion]
      overall_score: float
      total_correct: int
      total_questions: int

class KpDiagnosis(BaseModel):
      kp_id: str
      kp_title: str
      mastery_change: float = Field(description="掌握度变化量")
      error_type: str = ""
      detail: str = ""
      recommendation: str = ""


class Diagnosis(BaseModel):
    overall_score: float
    kp_diagnosis: list[KpDiagnosis] = Field(default_factory=list)
    next_priority: list[str] = Field(default_factory=list)
    summary: str = ""

class LearningState(TypedDict, total=False):
      subject_id: str
      subject_name: str
      subject_type: str
      time_minutes: int
      plan: dict
      tutor_results: list[dict]
      quiz_data: dict
      user_answers: dict
      graded: dict
      diagnosis: dict
      session_done: bool