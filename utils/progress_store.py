"""
用户学习进度持久化存储。

将每个 session 的知识点掌握度、SM-2 状态持久化到 JSON 文件。
重启 App 后自动恢复学习进度。

Usage:
    from utils.progress_store import ProgressStore
    store = ProgressStore()
    store.save_progress(thread_id, diagnosis_result)
    progress = store.load_progress(thread_id)
"""

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)

# ---- 路径常量 ----
_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "user_progress.json"


class ProgressStore:
    """用户学习进度存储。

    以 thread_id 为 key，存储：
    - kp_progress: {kp_id: {mastery, ef, interval, repetitions, last_reviewed}}
    - sessions: [{date, subject, score, ...}]
    """

    def __init__(self, filepath: Path | None = None):
        self._filepath = filepath or _DEFAULT_PATH
        self._lock = threading.Lock()

    def load_progress(self, thread_id: str) -> dict[str, Any]:
        """加载指定 session 的学习进度。

        Returns:
            {
                "kp_progress": {kp_id: {...}, ...},
                "sessions": [...],
                "last_active": "2026-07-11",
            }
        """
        data = self._load_all()
        return data.get(thread_id, self._default_progress())

    def save_progress(
        self,
        thread_id: str,
        diagnosis: dict[str, Any],
        plan: dict[str, Any] | None = None,
    ) -> None:
        """保存学习进度（在 diagnose_node 后调用）。

        从诊断结果中提取每个知识点的掌握度变化，
        更新 SM-2 状态。

        Args:
            thread_id:  session ID
            diagnosis:  诊断结果 dict（来自 Diagnosis.model_dump()）
            plan:       学习计划 dict（可选，用于记录 session 信息）
        """
        progress = self.load_progress(thread_id)

        # 更新知识点进度
        for kp in diagnosis.get("kp_diagnosis", []):
            kp_id = kp["kp_id"]
            if kp_id not in progress["kp_progress"]:
                progress["kp_progress"][kp_id] = self._default_kp_progress()

            entry = progress["kp_progress"][kp_id]
            # 累加掌握度变化
            entry["mastery"] = max(0.0, min(1.0, entry.get("mastery", 0.0) + kp.get("mastery_change", 0)))
            entry["last_reviewed"] = date.today().isoformat()
            entry["error_type"] = kp.get("error_type", "")

        # 记录 session
        if plan:
            progress["sessions"].append({
                "date": date.today().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "subject": plan.get("subject_name", "unknown"),
                "score": diagnosis.get("overall_score", 0),
                "duration": plan.get("total_minutes", 0),
            })

        progress["last_active"] = date.today().isoformat()

        # 持久化
        self._save_all(thread_id, progress)
        logger.info(
            "progress_saved",
            thread_id=thread_id[:8],
            kp_count=len(progress["kp_progress"]),
            session_count=len(progress["sessions"]),
        )

    def get_mastered_ids(self, thread_id: str, threshold: float = 0.6) -> set[str]:
        """获取已掌握的知识点 ID 集合（mastery >= threshold）。"""
        progress = self.load_progress(thread_id)
        return {
            kp_id
            for kp_id, entry in progress["kp_progress"].items()
            if entry.get("mastery", 0) >= threshold
        }

    def get_due_reviews(self, thread_id: str) -> list[str]:
        """获取待复习的知识点 ID 列表。"""
        from utils.spaced_repetition import filter_due_reviews

        progress = self.load_progress(thread_id)
        kp_data = {}
        for kp_id, entry in progress["kp_progress"].items():
            kp_data[kp_id] = {
                "next_review": entry.get("last_reviewed", "2000-01-01"),
                "mastery": entry.get("mastery", 0),
            }
        return filter_due_reviews(kp_data)

    # ---- 内部方法 ----
    def _load_all(self) -> dict[str, Any]:
        """加载全部用户进度数据。"""
        if not self._filepath.exists():
            return {}
        try:
            with self._filepath.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("progress_load_error", error=str(e))
            return {}

    def _save_all(self, thread_id: str, progress: dict[str, Any]) -> None:
        """保存全部用户进度数据。"""
        with self._lock:
            data = self._load_all()
            data[thread_id] = progress
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            with self._filepath.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _default_progress() -> dict[str, Any]:
        return {
            "kp_progress": {},
            "sessions": [],
            "last_active": None,
        }

    @staticmethod
    def _default_kp_progress() -> dict[str, Any]:
        return {
            "mastery": 0.0,
            "ef": 2.5,
            "interval": 0,
            "repetitions": 0,
            "last_reviewed": None,
            "error_type": "",
        }
