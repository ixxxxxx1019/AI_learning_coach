"""
AI学习教练 —— Streamlit 前端界面

LangGraph 管理后端流程，Streamlit 负责 UI 交互。
st.session_state 在多次页面刷新间保持 Graph 状态。
"""

import uuid

import streamlit as st

from agent.graph import create_graph
from config.logging_config import get_logger, setup_logging
from utils.knowledge_graph import list_subjects, load_kg

# 全局初始化 structlog
setup_logging()
logger = get_logger(__name__)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="AI学习教练",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI学习教练")
st.caption("基于 LangGraph 的智能学习系统 —— 规划 → 讲解 → 测验 → 诊断")

# ============================================================
# 初始化 session_state —— Streamlit 的"持久内存"
# ============================================================
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
if "phase" not in st.session_state:
    st.session_state.phase = "setup"


# ============================================================
# Phase 1: 初始设置 —— 选学科、设时间
# ============================================================
if st.session_state.phase == "setup":
    kg = load_kg()
    subjects = list_subjects(kg)

    if not subjects:
        st.error("未找到学科数据，请检查 data/knowledge_graph.json")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        subject_names = [s["name"] for s in subjects]
        selected_name = st.selectbox("📖 选择学科", subject_names)
        selected = next(s for s in subjects if s["name"] == selected_name)

        st.metric("知识点总数", selected["kp_count"])
        st.metric("领域数", selected["domain_count"])

    with col2:
        time_minutes = st.slider("⏱ 学习时长（分钟）", 10, 60, 30, 5)
        if selected.get("description"):
            st.info(selected["description"])

    if st.button("🚀 开始学习", type="primary", use_container_width=True):
        with st.spinner("AI 正在制定学习计划..."):
            initial_state = {
                "subject_id": selected["id"],
                "subject_name": selected["name"],
                "subject_type": selected.get("type", "unknown"),
                "time_minutes": time_minutes,
            }

            logger.info(
                "session_start",
                subject=selected["id"],
                time=time_minutes,
            )

            result = st.session_state.graph.invoke(
                initial_state,
                st.session_state.config,
            )

            st.session_state.plan = result.get("plan", {})
            st.session_state.tutor_results = result.get("tutor_results", [])
            st.session_state.quiz_data = result.get("quiz_data", {})
            st.session_state.phase = "learning"

            logger.info(
                "phase_transition",
                from_phase="setup",
                to_phase="learning",
                tutor_phases=len(result.get("tutor_results", [])),
                quiz_count=len(result.get("quiz_data", {}).get("questions", [])),
            )

        st.rerun()


# ============================================================
# Phase 2: 展示讲解 + 测验答题
# ============================================================
elif st.session_state.phase in ("learning", "quiz"):
    plan = st.session_state.get("plan", {})
    tutor_results = st.session_state.get("tutor_results", [])

    # ---- 学习计划摘要 ----
    with st.expander("📋 学习计划概览", expanded=False):
        st.write(f"**{plan.get('subject_name')}** | 总时长: {plan.get('total_minutes')}分钟")
        st.caption(plan.get("rationale", ""))
        for p in plan.get("phases", []):
            st.write(f"- {p['name']} ({p['type']}, {p['estimated_minutes']}分钟)")

    # ---- AI 讲解 ----
    st.subheader("📖 AI 讲解")
    for result in tutor_results:
        with st.expander(
            f"{result['phase_name']} ({result['phase_type']})",
            expanded=(len(tutor_results) <= 3),
        ):
            st.markdown(result["content"])

    st.divider()

    # ---- 测验题 ----
    st.subheader("📝 随堂测验")
    st.caption("完课后请作答，检验学习效果")

    quiz_data = st.session_state.get("quiz_data", {})
    questions = quiz_data.get("questions", [])
    st.session_state.phase = "quiz"

    for q in questions:
        qid = q["id"]
        key = f"answer_{qid}"

        if q["type"] == "multiple_choice":
            options = q.get("options", [])
            st.session_state[key] = st.radio(
                f"**{qid}.** {q['stem']}",
                options,
                index=None,
                key=f"radio_{qid}",
            )

        elif q["type"] == "fill_blank":
            st.session_state[key] = st.text_input(
                f"**{qid}.** {q['stem']}",
                key=f"fill_{qid}",
            )

        elif q["type"] == "translation":
            st.session_state[key] = st.text_area(
                f"**{qid}.** {q['stem']}",
                key=f"trans_{qid}",
                height=80,
            )

    if st.button("📤 提交答案", type="primary", use_container_width=True):
        with st.spinner("AI 正在批改..."):
            user_answers = {}
            for q in questions:
                qid = q["id"]
                user_answers[qid] = st.session_state.get(f"answer_{qid}", "") or ""

            logger.info("submitting_answers", answer_count=len(user_answers))

            st.session_state.graph.update_state(
                st.session_state.config,
                {"user_answers": user_answers},
            )

            result = st.session_state.graph.invoke(
                None,
                st.session_state.config,
            )

            st.session_state.graded = result.get("graded", {})
            st.session_state.diagnosis = result.get("diagnosis", {})
            st.session_state.phase = "result"

        st.rerun()


# ============================================================
# Phase 3: 批改结果 + 学习诊断
# ============================================================
elif st.session_state.phase == "result":
    graded = st.session_state.get("graded", {})
    diagnosis = st.session_state.get("diagnosis", {})

    st.success("✅ 本轮学习完成！")

    # ---- 成绩总览 ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总分", f"{graded.get('overall_score', 0):.0f} 分")
    with col2:
        st.metric(
            "正确率",
            f"{graded.get('total_correct', 0)}/{graded.get('total_questions', 0)}",
        )
    with col3:
        st.metric("综合评分", f"{diagnosis.get('overall_score', 0):.0f} 分")
    with col4:
        st.metric("会话ID", st.session_state.get("thread_id", "N/A")[:8])

    st.divider()

    # ---- 逐题详情 ----
    st.subheader("📋 答题详情")
    for gq in graded.get("graded_questions", []):
        is_correct = gq.get("is_correct", False)
        icon = "✅" if is_correct else "❌"
        label = f"{icon} {gq['question_id']}: {gq.get('feedback', '')[:60]}..."
        with st.expander(label):
            st.write(f"**你的答案**: {gq.get('user_answer', '')}")
            st.write(f"**正确答案**: {gq.get('correct_answer', '')}")
            if not is_correct:
                st.error(f"错误分析: {gq.get('error_analysis', '暂无')}")
            st.info(f"评语: {gq.get('feedback', '暂无')}")

    st.divider()

    # ---- 学习诊断 ----
    st.subheader("🔍 学习诊断")
    st.info(diagnosis.get("summary", "暂无总结"))

    st.write("**知识点掌握度变化:**")
    for kp in diagnosis.get("kp_diagnosis", []):
        change = kp.get("mastery_change", 0)
        delta = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
        col_a, col_b = st.columns([1, 4])
        with col_a:
            st.metric(kp.get("kp_title", kp.get("kp_id", "")), delta)
        with col_b:
            st.caption(kp.get("recommendation", ""))

    next_priority = diagnosis.get("next_priority", [])
    if next_priority:
        st.write(f"**📌 下一步优先复习:** {', '.join(next_priority)}")

    st.divider()

    if st.button("🔄 开始新一轮学习", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
