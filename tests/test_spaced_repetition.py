"""SM-2 间隔重复算法测试。"""

from datetime import date, timedelta

from utils.spaced_repetition import (
    calculate_next_review,
    calculate_retention_probability,
    filter_due_reviews,
    sm2_update,
)


class TestSM2Update:
    """SM-2 算法核心逻辑测试。"""

    def test_first_correct(self):
        """首次学习答对：interval=1, rep=1, EF 略有上升。"""
        ef, interval, rep = sm2_update(2.5, 0, 0, 0.8)
        assert interval == 1
        assert rep == 1
        assert ef > 2.5  # 高分 → EF 上升

    def test_first_correct_boost_ef(self):
        """满分 → EF 显著上升。"""
        ef, _, _ = sm2_update(2.5, 0, 0, 1.0)
        assert ef > 2.5

    def test_second_correct(self):
        """第二次复习答对：interval=6。"""
        _ef, interval, rep = sm2_update(2.5, 1, 1, 0.8)
        assert interval == 6
        assert rep == 2

    def test_third_correct(self):
        """第三次及以上：interval = round(prev_interval * EF)。"""
        _ef, interval, rep = sm2_update(2.5, 7, 3, 0.9)
        assert interval == round(7 * 2.5)  # 18
        assert rep == 4

    def test_failed_resets(self):
        """答错（score < 0.6）：interval 重置为 1，repetitions 重置为 0。"""
        _ef, interval, rep = sm2_update(2.5, 30, 5, 0.3)
        assert interval == 1
        assert rep == 0

    def test_ef_lower_bound(self):
        """EF 不低于 1.3。"""
        ef = 1.3
        for _ in range(10):
            ef, _, _ = sm2_update(ef, 1, 0, 0.0)  # 连续零分
        assert ef >= 1.3

    def test_borderline_score(self):
        """刚好 0.6 分 → 及格（>= 0.6）。"""
        _, interval, rep = sm2_update(2.5, 0, 0, 0.6)
        assert interval == 1
        assert rep == 1

    def test_borderline_fail(self):
        """刚好 0.59 分 → 不及格（< 0.6）。"""
        _, interval, rep = sm2_update(2.5, 30, 5, 0.59)
        assert interval == 1
        assert rep == 0


class TestNextReview:
    """复习日期计算测试。"""

    def test_calculate_next_review(self):
        base = date(2026, 1, 1)
        next_date = calculate_next_review(7, base)
        assert next_date == "2026-01-08"

    def test_calculate_next_review_default(self):
        next_date = calculate_next_review(0)
        # interval=0 → 今天
        assert next_date == date.today().isoformat()


class TestFilterDueReviews:
    """待复习筛选测试。"""

    def test_filter_due(self):
        today = date.today()
        progress = {
            "k001": {"next_review": today.isoformat(), "mastery": 0.8},
            "k002": {"next_review": (today - timedelta(days=3)).isoformat(), "mastery": 0.4},
            "k003": {"next_review": (today + timedelta(days=7)).isoformat(), "mastery": 0.9},
        }
        due = filter_due_reviews(progress, today)
        # k002 过期最早 → 排第一, k001 今天到期 → 排第二
        assert due[0] == "k002"
        assert "k001" in due
        assert "k003" not in due  # 7 天后才到期


class TestRetentionProbability:
    """记忆保持概率测试。"""

    def test_no_elapse(self):
        p = calculate_retention_probability(2.5, 7, 0)
        assert p == 1.0

    def test_partial_decay(self):
        p = calculate_retention_probability(2.5, 7, 3)
        assert 0.5 < p < 1.0

    def test_full_decay(self):
        p = calculate_retention_probability(2.5, 7, 14)
        assert p < 0.7
