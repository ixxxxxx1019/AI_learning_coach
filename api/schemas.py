"""
API 请求/响应 Pydantic 模型。

与 agent/models.py 分开：前者定义 LLM 结构化输出，后者定义 REST API 契约。
"""

from pydantic import BaseModel, Field

# ============================================================
# 请求模型
# ============================================================


class CreateSessionRequest(BaseModel):
    """创建学习 session 请求。"""

    subject_id: str = Field(..., description="学科 ID，如 cet6_vocab", examples=["cet6_vocab"])
    time_minutes: int = Field(
        default=20, ge=5, le=120, description="学习时长（分钟）", examples=[20]
    )


class SubmitAnswersRequest(BaseModel):
    """提交答案请求。"""

    answers: dict[str, str] = Field(
        ...,
        description="题目 ID → 用户答案 映射",
        examples=[{"q1": "B. complex", "q2": "He is a sophisticated man"}],
    )


# ============================================================
# 响应模型
# ============================================================


class SubjectInfo(BaseModel):
    """学科基本信息。"""

    id: str
    name: str
    description: str = ""
    type: str = "unknown"
    domain_count: int = 0
    kp_count: int = 0


class KnowledgePointInfo(BaseModel):
    """知识点摘要。"""

    id: str
    title: str
    type: str = ""
    difficulty: int = 1
    domain_name: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PhaseInfo(BaseModel):
    """学习阶段信息。"""

    index: int
    name: str
    type: str
    kp_ids: list[str]
    estimated_minutes: int
    instruction: str = ""


class PlanInfo(BaseModel):
    """学习计划。"""

    subject_name: str
    total_minutes: int
    rationale: str = ""
    phases: list[dict] = Field(default_factory=list)


class TutorResult(BaseModel):
    """单阶段教学内容。"""

    phase_index: int
    phase_name: str
    phase_type: str
    content: str


class QuestionInfo(BaseModel):
    """测验题。"""

    id: str
    type: str
    stem: str
    options: list[str] = Field(default_factory=list)
    target_kp_id: str = ""
    target_kp_title: str = ""
    difficulty: str = "medium"


class QuizInfo(BaseModel):
    """测验。"""

    questions: list[dict] = Field(default_factory=list)
    estimated_total_minutes: int = 0


class SessionCreatedResponse(BaseModel):
    """创建 session 响应（Phase 1 完成）。"""

    session_id: str
    subject_name: str = ""
    plan: dict | None = None
    tutor_content: list[dict] = Field(default_factory=list)
    quiz: dict | None = None


class GradedQuestionInfo(BaseModel):
    """单题批改结果。"""

    question_id: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    error_analysis: str = ""
    feedback: str = ""
    target_kp_id: str = ""


class GradingInfo(BaseModel):
    """批改结果。"""

    graded_questions: list[dict] = Field(default_factory=list)
    overall_score: float = 0.0
    total_correct: int = 0
    total_questions: int = 0


class DiagnosisInfo(BaseModel):
    """诊断结果。"""

    overall_score: float = 0.0
    kp_diagnosis: list[dict] = Field(default_factory=list)
    next_priority: list[str] = Field(default_factory=list)
    summary: str = ""


class SubmitAnswersResponse(BaseModel):
    """提交答案响应（Phase 2 完成）。"""

    session_id: str
    graded: dict | None = None
    diagnosis: dict | None = None
    session_done: bool = True


class SessionStatusResponse(BaseModel):
    """Session 状态查询响应。"""

    session_id: str
    phase: str  # "quiz" | "result"
    plan: dict | None = None
    quiz: dict | None = None
    graded: dict | None = None
    diagnosis: dict | None = None
    session_done: bool = False


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    checks: dict
