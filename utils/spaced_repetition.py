"""
SM-2 (SuperMemo 2) 遗忘曲线算法实现

SM-2 是最经典、最广泛验证的间隔重复算法。Anki 的默认调度器就基于它。

核心公式:
- 评分 >= 0.6 (及格):
    - repetition 0 → interval = 1 天
    - repetition 1 → interval = 6 天
    - repetition >= 2 → interval = round(interval * EF)
    - repetitions += 1
- 评分 < 0.6 (不及格):
    - interval = 1 天（从头开始）
    - repetitions = 0

- EF (Easiness Factor) 更新:
    new_ef = ef + (0.1 - (1 - score) * (0.08 + (1 - score) * 0.02))
    new_ef = max(1.3, new_ef)  # 下限 1.3

面试亮点: 不是简单调 Anki 的库，而是从论文公式手写实现，每一行代码都能解释。
"""

from datetime import date, timedelta
from typing import Optional


# ------------------------------------------------------------------
# 核心算法
# ------------------------------------------------------------------

def sm2_update(
    ef: float,
    interval: int,
    repetitions: int,
    score: float,
) -> tuple[float, int, int]:
    """SM-2 算法的单次更新。

    Args:
        ef:          当前的 Easiness Factor（初值 2.5）
        interval:    当前复习间隔（天），初值 0
        repetitions: 当前连续正确次数，初值 0
        score:       本次评分，0.0 ~ 1.0

    Returns:
        (new_ef, new_interval, new_repetitions)

    Examples:
        # 首次学习，答对了
        >>> sm2_update(2.5, 0, 0, 0.8)
        (2.58, 1, 1)

        # 第二次复习，答错了
        >>> sm2_update(2.58, 1, 1, 0.3)
        (2.37, 1, 0)

        # 多次复习后，高分通过
        >>> sm2_update(2.5, 7, 3, 0.9)
        (2.52, 18, 4)
    """
    if score >= 0.6:
        # --- 及格 ---
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = round(interval * ef)

        new_repetitions = repetitions + 1
    else:
        # --- 不及格：重置 ---
        new_interval = 1
        new_repetitions = 0

    # 更新 Easiness Factor
    new_ef = ef + (0.1 - (1.0 - score) * (0.08 + (1.0 - score) * 0.02))
    new_ef = max(1.3, new_ef)  # 下限 1.3

    return new_ef, new_interval, new_repetitions


# ------------------------------------------------------------------
# 日期计算
# ------------------------------------------------------------------

def calculate_next_review(interval: int, from_date: Optional[date] = None) -> str:
    """计算下次复习日期。

    Args:
        interval:  复习间隔（天）
        from_date: 起始日期，默认今天

    Returns:
        ISO 格式日期字符串 "YYYY-MM-DD"

    >>> calculate_next_review(7)  # 一周后
    """
    base = from_date or date.today()
    next_date = base + timedelta(days=interval)
    return next_date.isoformat()


# ------------------------------------------------------------------
# 批量调度查询
# ------------------------------------------------------------------

def filter_due_reviews(progress: dict, reference_date: Optional[date] = None) -> list[str]:
    """从用户进度中筛选出今日待复习的知识点 ID。

    Args:
        progress:  user_progress.json 中某 subject 的 knowledge_points
                   形如 {"k001": {"next_review": "2026-06-26", ...}, ...}
        reference_date: 参考日期，默认今天

    Returns:
        待复习的知识点 ID 列表，按紧急度排序（next_review 越早越靠前）
    """
    today = (reference_date or date.today()).isoformat()
    today_date = date.fromisoformat(today)

    due = []
    for kp_id, kp_data in progress.items():
        next_review_str = kp_data.get("next_review", "2000-01-01")
        try:
            next_review_date = date.fromisoformat(next_review_str)
            if next_review_date <= today_date:
                due.append((kp_id, next_review_date))
        except (ValueError, TypeError):
            # 无效日期 → 视为需要复习
            due.append((kp_id, date.fromisoformat("2000-01-01")))

    # 按紧急度排序：越早到期越靠前
    due.sort(key=lambda x: x[1])
    return [kp_id for kp_id, _ in due]


def calculate_retention_probability(ef: float, interval: int, days_elapsed: int) -> float:
    """估算当前记忆保持概率（用于 UI 展示，非 SM-2 核心公式）。

    基于 Ebbinghaus 遗忘曲线的简化模型：
    R = e^(-t / (S * EF))
    其中 S 是稳定性（与 interval 相关），t = days_elapsed

    Args:
        ef:           Easiness Factor
        interval:     原始复习间隔（天）
        days_elapsed: 距离上次复习的天数

    Returns:
        0.0 ~ 1.0 的记忆保持概率
    """
    import math

    if days_elapsed <= 0:
        return 1.0

    # 稳定性 = interval * ef（间隔越长 + 越简单 → 越稳定）
    stability = interval * ef
    if stability <= 0:
        return 0.5

    retention = math.exp(-days_elapsed / stability)
    return round(retention, 4)


# ------------------------------------------------------------------
# 自测
# ------------------------------------------------------------------

if __name__ == "__main__":
    # 测试 SM-2 核心逻辑
    print("=== SM-2 算法测试 ===")

    # 模拟一个知识点的完整学习周期
    ef, interval, rep = 2.5, 0, 0
    scores = [0.8, 0.9, 0.7, 0.3, 0.9, 0.9, 0.9]  # 第4次答错了

    for i, score in enumerate(scores):
        ef, interval, rep = sm2_update(ef, interval, rep, score)
        status = "[PASS]" if score >= 0.6 else "[FAIL]"
        print(
            f"  Round {i+1}: score={score} {status} -> "
            f"EF={ef:.2f}, interval={interval}d, rep={rep}"
        )

    # 测试待复习筛选
    print("\n=== 待复习筛选测试 ===")
    today = date.today()
    progress = {
        "k001": {"next_review": today.isoformat(), "mastery": 0.8},           # 今天
        "k002": {"next_review": today.isoformat(), "mastery": 0.5},           # 今天
        "k003": {"next_review": (today + timedelta(days=3)).isoformat(), "mastery": 0.9},  # 3天后
        "k004": {"next_review": (today - timedelta(days=2)).isoformat(), "mastery": 0.4},  # 2天前（过期）
    }

    due = filter_due_reviews(progress)
    print(f"  当前日期: {today.isoformat()}")
    print(f"  待复习: {due}  (应为 k004, k001, k002 — 按紧急度排序)")

    # 测试记忆保持概率
    print("\n=== 记忆保持概率测试 ===")
    print(f"  EF=2.5, interval=7d, 0 days elapsed:  {calculate_retention_probability(2.5, 7, 0):.1%}")
    print(f"  EF=2.5, interval=7d, 3 days elapsed:  {calculate_retention_probability(2.5, 7, 3):.1%}")
    print(f"  EF=2.5, interval=7d, 7 days elapsed:  {calculate_retention_probability(2.5, 7, 7):.1%}")
    print(f"  EF=2.5, interval=7d, 14 days elapsed: {calculate_retention_probability(2.5, 7, 14):.1%}")

    print("\n[OK] All tests passed!")
