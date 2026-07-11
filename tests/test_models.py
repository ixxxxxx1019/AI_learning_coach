"""Pydantic 模型序列化/反序列化测试。"""

from agent.models import (
    Diagnosis,
    GradedQuestion,
    GradingResult,
    KpDiagnosis,
    Question,
    Quiz,
    StudyPhase,
    StudyPlan,
)


class TestStudyPlan:
    """学习计划模型测试。"""

    def test_study_plan_creation(self):
        plan = StudyPlan(
            subject_name="CET6词汇",
            total_minutes=30,
            rationale="测试计划",
            phases=[
                StudyPhase(
                    type="review",
                    name="复习基础",
                    kp_ids=["k001"],
                    estimated_minutes=5,
                    instruction="复习之前学过的内容",
                )
            ],
        )
        assert plan.subject_name == "CET6词汇"
        assert len(plan.phases) == 1

    def test_study_plan_serialization(self):
        plan = StudyPlan(
            subject_name="Python基础",
            total_minutes=20,
            rationale="测试",
            phases=[],
        )
        data = plan.model_dump()
        assert data["subject_name"] == "Python基础"
        # 可往返
        plan2 = StudyPlan(**data)
        assert plan2.total_minutes == 20


class TestQuiz:
    """测验题模型测试。"""

    def test_question_creation(self):
        q = Question(
            id="q1",
            type="multiple_choice",
            stem="What does 'hello' mean?",
            options=["A. 你好", "B. 再见", "C. 谢谢", "D. 对不起"],
            correct="A. 你好",
            explanation="hello means 你好",
            target_kp_id="k001",
            target_kp_title="basic greeting",
        )
        assert q.type == "multiple_choice"
        assert len(q.options) == 4

    def test_quiz_creation(self):
        quiz = Quiz(
            questions=[
                Question(
                    id="q1",
                    type="fill_blank",
                    stem="Fill in: __ world",
                    correct="hello",
                    explanation="Common greeting",
                    target_kp_id="k001",
                    target_kp_title="greeting",
                )
            ],
            estimated_total_minutes=10,
        )
        assert quiz.estimated_total_minutes == 10
        assert len(quiz.questions) == 1


class TestGradingResult:
    """批改结果模型测试。"""

    def test_grading_result(self):
        result = GradingResult(
            graded_questions=[
                GradedQuestion(
                    question_id="q1",
                    user_answer="A",
                    correct_answer="A",
                    is_correct=True,
                    error_analysis="",
                    feedback="Great job!",
                    target_kp_id="k001",
                )
            ],
            overall_score=100.0,
            total_correct=1,
            total_questions=1,
        )
        assert result.overall_score == 100.0
        assert result.graded_questions[0].is_correct


class TestDiagnosis:
    """学习诊断模型测试。"""

    def test_diagnosis(self):
        diag = Diagnosis(
            overall_score=85.0,
            kp_diagnosis=[
                KpDiagnosis(
                    kp_id="k001",
                    kp_title="变量",
                    mastery_change=0.15,
                    error_type="",
                    detail="掌握良好",
                    recommendation="继续深入学习",
                )
            ],
            next_priority=["k002"],
            summary="整体表现良好",
        )
        assert diag.overall_score == 85.0
        assert abs(diag.kp_diagnosis[0].mastery_change - 0.15) < 0.01
        assert "k002" in diag.next_priority
