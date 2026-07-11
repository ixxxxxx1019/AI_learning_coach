"""知识图谱工具函数测试。"""

import pytest

from utils.knowledge_graph import (
    find_weak_root_causes,
    get_dependents,
    get_kp,
    get_learnable_kps,
    get_prerequisites,
    list_subjects,
)


# 共享的测试用知识图谱
@pytest.fixture
def sample_kg():
    return {
        "subjects": [
            {
                "id": "test_vocab",
                "name": "测试词汇",
                "type": "vocabulary",
                "domains": [
                    {
                        "id": "d1",
                        "name": "测试领域",
                        "knowledge_points": [
                            {
                                "id": "k001",
                                "title": "basic",
                                "type": "word",
                                "content": {"word": "basic"},
                                "difficulty": 1,
                                "prerequisites": [],
                                "tags": ["基础"],
                            },
                            {
                                "id": "k002",
                                "title": "advanced",
                                "type": "word",
                                "content": {"word": "advanced"},
                                "difficulty": 4,
                                "prerequisites": ["k001"],
                                "tags": ["高频"],
                            },
                            {
                                "id": "k003",
                                "title": "expert",
                                "type": "word",
                                "content": {"word": "expert"},
                                "difficulty": 5,
                                "prerequisites": ["k002"],
                                "tags": ["高频"],
                            },
                        ],
                    }
                ],
            }
        ]
    }


class TestKnowledgeGraphBasics:
    """基础查询测试。"""

    def test_list_subjects(self, sample_kg):
        subjects = list_subjects(sample_kg)
        assert len(subjects) == 1
        assert subjects[0]["name"] == "测试词汇"
        assert subjects[0]["kp_count"] == 3

    def test_get_kp_found(self, sample_kg):
        kp = get_kp(sample_kg, "k001")
        assert kp is not None
        assert kp["title"] == "basic"

    def test_get_kp_not_found(self, sample_kg):
        kp = get_kp(sample_kg, "k999")
        assert kp is None

    def test_get_kp_includes_context(self, sample_kg):
        kp = get_kp(sample_kg, "k001")
        assert kp["_subject_name"] == "测试词汇"
        assert kp["_domain_name"] == "测试领域"


class TestDependencyQueries:
    """依赖图查询测试。"""

    def test_prerequisites(self, sample_kg):
        prereqs = get_prerequisites(sample_kg, "k002")
        assert len(prereqs) == 1
        assert prereqs[0]["title"] == "basic"

    def test_no_prerequisites(self, sample_kg):
        prereqs = get_prerequisites(sample_kg, "k001")
        assert len(prereqs) == 0

    def test_dependents(self, sample_kg):
        deps = get_dependents(sample_kg, "k001")
        assert len(deps) == 1
        assert deps[0]["title"] == "advanced"


class TestWeakRootCause:
    """薄弱点根因分析测试。"""

    def test_root_cause_simple(self, sample_kg):
        roots = find_weak_root_causes(sample_kg, ["k002"])
        # k002 依赖 k001，但 k001 不在 weak 列表中
        # k002 是 hard (difficulty >= 3)，所以 k001 会被标记为 weak prerequisite
        # 但 k001 不在 weak_set 且 difficulty=1 < 3，走 else 分支
        # 所以 k002 是根因
        root_ids = [r["id"] for r in roots]
        assert "k002" in root_ids

    def test_root_cause_deep_chain(self, sample_kg):
        roots = find_weak_root_causes(sample_kg, ["k003"])
        # k003 依赖 k002 (difficulty=4 >= 3), k002 依赖 k001 (difficulty=1 < 3)
        # k002 在 weak_set 吗？不在
        # 递归：k002 的 prereqs 中，k001 difficulty=1 < 3，所以 k002 是根因
        root_ids = [r["id"] for r in roots]
        assert len(root_ids) > 0


class TestLearnableKps:
    """可学习知识点推荐测试。"""

    def test_learnable_empty_mastered(self, sample_kg):
        learnable = get_learnable_kps(sample_kg, "test_vocab", mastered_ids=set(), count=5)
        # 只有 k001 没有前置依赖
        ids = [kp["id"] for kp in learnable]
        assert "k001" in ids
        assert "k002" not in ids  # k002 依赖 k001

    def test_learnable_after_mastering(self, sample_kg):
        learnable = get_learnable_kps(sample_kg, "test_vocab", mastered_ids={"k001"}, count=5)
        ids = [kp["id"] for kp in learnable]
        assert "k002" in ids  # k001 已掌握，k002 可学
