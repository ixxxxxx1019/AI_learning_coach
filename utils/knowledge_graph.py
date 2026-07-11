"""
通用知识图谱加载与查询工具

设计原则:
1. 不绑定特定学科 —— 任何 subject type 共用同一套查询 API
2. knowledge_points[].content 是灵活的自由 JSON，不同学科有不同结构
3. 支持依赖图查询（前置/后置依赖、学习路径拓扑排序）
"""

import json
from pathlib import Path

# ------------------------------------------------------------------
# 路径常量
# ------------------------------------------------------------------

_DEFAULT_KG_PATH = Path(__file__).parent.parent / "data" / "knowledge_graph.json"


# ------------------------------------------------------------------
# 加载
# ------------------------------------------------------------------


def load_kg(path: Path | None = None) -> dict:
    """加载知识图谱 JSON 文件。

    Args:
        path: 可选的文件路径，默认 data/knowledge_graph.json

    Returns:
        完整的知识图谱 dict
    """
    filepath = path or _DEFAULT_KG_PATH
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def save_kg(kg: dict, path: Path | None = None):
    """保存知识图谱到 JSON 文件。"""
    filepath = path or _DEFAULT_KG_PATH
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# Subject 查询
# ------------------------------------------------------------------


def list_subjects(kg: dict) -> list[dict]:
    """列出所有学科的基本信息。

    Returns:
        [{"id": "cet6_vocab", "name": "CET6英语词汇", "type": "vocabulary", ...}, ...]
    """
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "description": s.get("description", ""),
            "type": s.get("type", "unknown"),
            "domain_count": len(s.get("domains", [])),
            "kp_count": sum(len(d.get("knowledge_points", [])) for d in s.get("domains", [])),
        }
        for s in kg.get("subjects", [])
    ]


def get_subject(kg: dict, subject_id: str) -> dict | None:
    """按 ID 获取学科完整数据。"""
    for s in kg.get("subjects", []):
        if s["id"] == subject_id:
            return s
    return None


# ------------------------------------------------------------------
# Domain 查询
# ------------------------------------------------------------------


def list_domains(kg: dict, subject_id: str) -> list[dict]:
    """列出某学科下的所有领域。"""
    subject = get_subject(kg, subject_id)
    if not subject:
        return []
    return [
        {
            "id": d["id"],
            "name": d["name"],
            "description": d.get("description", ""),
            "kp_count": len(d.get("knowledge_points", [])),
        }
        for d in subject.get("domains", [])
    ]


# ------------------------------------------------------------------
# Knowledge Point 查询
# ------------------------------------------------------------------


def get_kp(kg: dict, kp_id: str) -> dict | None:
    """按 ID 跨所有 subject 查找知识点。"""
    for s in kg.get("subjects", []):
        for d in s.get("domains", []):
            for kp in d.get("knowledge_points", []):
                if kp["id"] == kp_id:
                    # 附加上下文信息
                    result = dict(kp)
                    result["_subject_id"] = s["id"]
                    result["_subject_name"] = s["name"]
                    result["_subject_type"] = s.get("type", "")
                    result["_domain_id"] = d["id"]
                    result["_domain_name"] = d["name"]
                    return result
    return None


def list_all_kps(kg: dict, subject_id: str | None = None) -> list[dict]:
    """列出所有知识点（可选按 subject 过滤）。

    Returns:
        平铺的知识点列表，每个都附带 _subject_id, _domain_id 等上下文
    """
    results = []
    for s in kg.get("subjects", []):
        if subject_id and s["id"] != subject_id:
            continue
        for d in s.get("domains", []):
            for kp in d.get("knowledge_points", []):
                item = dict(kp)
                item["_subject_id"] = s["id"]
                item["_subject_name"] = s["name"]
                item["_subject_type"] = s.get("type", "")
                item["_domain_id"] = d["id"]
                item["_domain_name"] = d["name"]
                results.append(item)
    return results


# ------------------------------------------------------------------
# 依赖图查询（核心亮点！）
# ------------------------------------------------------------------


def get_prerequisites(kg: dict, kp_id: str) -> list[dict]:
    """获取某个知识点的所有前置依赖（递归一层）。

    这是面试中"知识点依赖图薄弱点定位"的技术基础。
    要理解 kp_id，必须先掌握返回的这些知识点。
    """
    kp = get_kp(kg, kp_id)
    if not kp:
        return []

    prereq_ids = kp.get("prerequisites", [])
    return [get_kp(kg, pid) for pid in prereq_ids if get_kp(kg, pid)]


def get_dependents(kg: dict, kp_id: str) -> list[dict]:
    """获取依赖某个知识点的所有后续知识点（谁依赖它）。

    用于判断：如果这个知识点没掌握，哪些后续学习会受影响。
    """
    all_kps = list_all_kps(kg)
    return [kp for kp in all_kps if kp_id in kp.get("prerequisites", [])]


def find_weak_root_causes(
    kg: dict,
    weak_kp_ids: list[str],
) -> list[dict]:
    """找出薄弱点的根因知识点。

    核心算法：给定一组"薄弱"知识点（用户答错的），
    回溯它们的依赖链，找出最上游的未掌握前置知识点。
    这些根因才是真正需要补习的地方。

    面试亮点: "不是简单地标记哪些错了，而是追溯到知识依赖链的根节点"

    Args:
        kg:          知识图谱
        weak_kp_ids: 用户答错的知识点 ID 列表

    Returns:
        根因知识点列表（最上游的未掌握前置）

    Example:
        如果 k102 依赖 k101, k101 依赖 k100，
        用户答错了 k102，根因可能是 k100 没掌握。
    """
    weak_set = set(weak_kp_ids)
    root_causes = []

    for wid in weak_kp_ids:
        prereqs = get_prerequisites(kg, wid)
        # 检查是否有前置知识薄弱
        weak_prereqs = [p for p in prereqs if p["id"] in weak_set or p.get("difficulty", 5) >= 3]
        if weak_prereqs:
            # 递归找根因：如果前置也在薄弱列表中，继续向上追溯
            for wp in weak_prereqs:
                root_causes.extend(find_weak_root_causes(kg, [wp["id"]]))
        else:
            # 没有薄弱前置 → 这个就是根因
            kp = get_kp(kg, wid)
            if kp:
                root_causes.append(kp)

    # 去重
    seen = set()
    unique = []
    for r in root_causes:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    return unique


# ------------------------------------------------------------------
# 筛选与排序
# ------------------------------------------------------------------


def filter_kps(
    kg: dict,
    subject_id: str | None = None,
    domain_id: str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """按条件筛选知识点。"""
    all_kps = list_all_kps(kg, subject_id=subject_id)

    results = []
    for kp in all_kps:
        if domain_id and kp.get("_domain_id") != domain_id:
            continue
        diff = kp.get("difficulty", 5)
        if min_difficulty is not None and diff < min_difficulty:
            continue
        if max_difficulty is not None and diff > max_difficulty:
            continue
        if tags:
            kp_tags = set(kp.get("tags", []))
            if not kp_tags.intersection(tags):
                continue
        results.append(kp)

    return results


def get_learnable_kps(
    kg: dict,
    subject_id: str,
    mastered_ids: set[str],
    count: int = 5,
) -> list[dict]:
    """获取可学习的新知识点（前置依赖已满足）。

    Args:
        kg:           知识图谱
        subject_id:   目标学科 ID
        mastered_ids: 已掌握的知识点 ID 集合（mastery >= 0.6）
        count:        返回数量上限

    Returns:
        可学习的知识点列表，按难度升序（简单优先）
    """
    all_kps = list_all_kps(kg, subject_id=subject_id)
    # 排除已掌握的和已在进度中的
    available = [kp for kp in all_kps if kp["id"] not in mastered_ids]
    # 筛选前置依赖已满足的
    learnable = [
        kp
        for kp in available
        if all(pid in mastered_ids or not get_kp(kg, pid) for pid in kp.get("prerequisites", []))
    ]
    # 按难度升序
    learnable.sort(key=lambda k: k.get("difficulty", 5))
    return learnable[:count]


# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------


def get_kp_count(kg: dict, subject_id: str | None = None) -> int:
    """统计知识点总数。"""
    return len(list_all_kps(kg, subject_id=subject_id))


def get_total_subjects(kg: dict) -> int:
    """统计学科总数。"""
    return len(kg.get("subjects", []))


# ------------------------------------------------------------------
# 自测
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 知识图谱工具测试 ===\n")

    # 用示例数据测试
    sample_kg = {
        "subjects": [
            {
                "id": "cet6_vocab",
                "name": "CET6词汇",
                "type": "vocabulary",
                "domains": [
                    {
                        "id": "academic",
                        "name": "学术类",
                        "knowledge_points": [
                            {
                                "id": "k001",
                                "title": "simple",
                                "type": "word",
                                "content": {"word": "simple", "definition_cn": "简单的"},
                                "difficulty": 1,
                                "prerequisites": [],
                                "tags": ["基础"],
                            },
                            {
                                "id": "k002",
                                "title": "sophisticated",
                                "type": "word",
                                "content": {"word": "sophisticated", "definition_cn": "复杂的"},
                                "difficulty": 4,
                                "prerequisites": ["k001"],
                                "tags": ["高频"],
                            },
                        ],
                    }
                ],
            }
        ]
    }

    # 测试 list_subjects
    subjects = list_subjects(sample_kg)
    print(f"1. 学科列表: {[s['name'] for s in subjects]}")

    # 测试 get_kp
    kp = get_kp(sample_kg, "k002")
    print(f"2. 查询 k002: {kp['title']} (subject={kp['_subject_name']})")

    # 测试 get_prerequisites
    prereqs = get_prerequisites(sample_kg, "k002")
    print(f"3. k002 的前置依赖: {[p['title'] for p in prereqs]}")

    # 测试 get_dependents
    deps = get_dependents(sample_kg, "k001")
    print(f"4. 依赖 k001 的知识点: {[d['title'] for d in deps]}")

    # 测试 find_weak_root_causes
    roots = find_weak_root_causes(sample_kg, ["k002"])
    print(f"5. k002 薄弱的根因: {[r['title'] for r in roots]}")

    # 测试 get_learnable_kps
    learnable = get_learnable_kps(sample_kg, "cet6_vocab", mastered_ids=set(), count=5)
    print(f"6. 可学习的新知识点: {[kp['title'] for kp in learnable]}")

    # 测试 get_learnable_kps with k001 mastered
    learnable2 = get_learnable_kps(sample_kg, "cet6_vocab", mastered_ids={"k001"}, count=5)
    print(f"7. 掌握k001后可学: {[kp['title'] for kp in learnable2]}")

    print("\n[OK] All knowledge graph tests passed!")
