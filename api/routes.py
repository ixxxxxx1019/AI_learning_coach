"""
API 路由定义。

所有 /api/* 端点在此集中管理。
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from api.deps import get_graph, get_kg
from api.schemas import (
    CreateSessionRequest,
    HealthResponse,
    KnowledgePointInfo,
    SessionCreatedResponse,
    SessionStatusResponse,
    SubjectInfo,
    SubmitAnswersRequest,
    SubmitAnswersResponse,
)
from config.logging_config import get_logger
from health import health_check
from utils.knowledge_graph import list_all_kps, list_subjects

logger = get_logger(__name__)

router = APIRouter(prefix="/api")

# ---- 内存 session 存储（用于快速查询 session 状态） ----
# key: session_id (thread_id), value: {"phase": "quiz"|"result", ...}
_sessions: dict[str, dict[str, Any]] = {}


# ============================================================
# Health
# ============================================================


@router.get("/health", response_model=HealthResponse)
def api_health():
    """健康检查 — 验证所有依赖正常。"""
    result = health_check()
    return HealthResponse(**result)


# ============================================================
# Subjects
# ============================================================


@router.get("/subjects", response_model=list[SubjectInfo])
def list_subjects_api():
    """列出所有可用学科。"""
    kg = get_kg()
    subjects = list_subjects(kg)
    return [SubjectInfo(**s) for s in subjects]


@router.get("/subjects/{subject_id}/kps", response_model=list[KnowledgePointInfo])
def list_kps_api(subject_id: str):
    """列出某学科的所有知识点。"""
    kg = get_kg()
    all_kps = list_all_kps(kg, subject_id=subject_id)
    return [
        KnowledgePointInfo(
            id=kp["id"],
            title=kp["title"],
            type=kp.get("type", ""),
            difficulty=kp.get("difficulty", 1),
            domain_name=kp.get("_domain_name", ""),
            prerequisites=kp.get("prerequisites", []),
            tags=kp.get("tags", []),
        )
        for kp in all_kps
    ]


# ============================================================
# Sessions
# ============================================================


@router.post("/sessions", response_model=SessionCreatedResponse, status_code=201)
def create_session(req: CreateSessionRequest):
    """创建学习 session。

    执行 LangGraph Phase 1：planner → tutor → quiz，
    在 grade 前暂停，返回教学内容和测验题。

    Args:
        req: 包含 subject_id 和 time_minutes

    Returns:
        session_id + plan + tutor_content + quiz
    """
    graph = get_graph()
    kg = get_kg()

    # 验证学科存在
    subjects = list_subjects(kg)
    subject = next((s for s in subjects if s["id"] == req.subject_id), None)
    if not subject:
        raise HTTPException(status_code=404, detail=f"Subject not found: {req.subject_id}")

    session_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "subject_id": req.subject_id,
        "subject_name": subject["name"],
        "subject_type": subject.get("type", "unknown"),
        "time_minutes": req.time_minutes,
    }

    logger.info("api_session_create", session_id=session_id, subject=req.subject_id)

    try:
        result = graph.invoke(initial_state, config)
    except Exception as e:
        logger.error("api_session_create_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"AI service error: {e}") from e

    # 缓存 session 状态
    _sessions[session_id] = {
        "phase": "quiz",
        "plan": result.get("plan"),
        "quiz": result.get("quiz_data"),
    }

    return SessionCreatedResponse(
        session_id=session_id,
        subject_name=result.get("subject_name", ""),
        plan=result.get("plan"),
        tutor_content=result.get("tutor_results", []),
        quiz=result.get("quiz_data"),
    )


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
def get_session(session_id: str):
    """查询 session 状态。

    返回当前阶段和已有数据（plan / quiz / graded / diagnosis）。
    """
    cached = _sessions.get(session_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    config = {"configurable": {"thread_id": session_id}}
    graph = get_graph()

    try:
        state = graph.get_state(config)
        full_state = state.values if state.values else {}
    except Exception:
        full_state = {}

    return SessionStatusResponse(
        session_id=session_id,
        phase=cached.get("phase", "unknown"),
        plan=cached.get("plan"),
        quiz=cached.get("quiz"),
        graded=cached.get("graded"),
        diagnosis=cached.get("diagnosis"),
        session_done=full_state.get("session_done", False),
    )


@router.post("/sessions/{session_id}/answers", response_model=SubmitAnswersResponse)
def submit_answers(session_id: str, req: SubmitAnswersRequest):
    """提交测验答案。

    执行 LangGraph Phase 2：grade → diagnose，
    返回批改结果和学习诊断。

    Args:
        session_id: 由 POST /sessions 返回的 session ID
        req:       qid → 用户答案 映射

    Returns:
        graded + diagnosis
    """
    cached = _sessions.get(session_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if cached.get("phase") != "quiz":
        raise HTTPException(status_code=400, detail="Answers already submitted for this session")

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    logger.info("api_submit_answers", session_id=session_id, answer_count=len(req.answers))

    try:
        graph.update_state(config, {"user_answers": req.answers})
        result = graph.invoke(None, config)
    except Exception as e:
        logger.error("api_submit_answers_failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"AI service error: {e}") from e

    # 更新缓存
    cached["phase"] = "result"
    cached["graded"] = result.get("graded")
    cached["diagnosis"] = result.get("diagnosis")

    return SubmitAnswersResponse(
        session_id=session_id,
        graded=result.get("graded"),
        diagnosis=result.get("diagnosis"),
        session_done=result.get("session_done", True),
    )


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """删除 session（清除缓存）。"""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
