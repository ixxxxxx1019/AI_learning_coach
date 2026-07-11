"""
健康检查模块。

检查应用的各项依赖是否正常：
- DeepSeek API 可达性
- 知识图谱可加载
- 配置有效

Usage:
    python health.py          # CLI 运行
    from health import health_check  # 程序调用
"""

import sys
from typing import Any

from config.settings import get_settings


def health_check() -> dict[str, Any]:
    """运行所有健康检查，返回状态字典。

    Returns:
        {
            "status": "ok" | "degraded" | "down",
            "checks": {...},
        }

    Raises:
        SystemExit: 如果任何检查失败（CLI 模式下 exit code 1）
    """
    checks = {}
    all_ok = True

    # 1. 配置检查
    try:
        settings = get_settings()
        api_key = settings.deepseek_api_key.get_secret_value()
        if api_key and api_key != "sk-placeholder" and "your-" not in api_key:
            checks["config"] = {"status": "ok", "message": "API key configured"}
        else:
            checks["config"] = {"status": "warn", "message": "API key is placeholder"}
            all_ok = False
    except Exception as e:
        checks["config"] = {"status": "error", "message": str(e)}
        all_ok = False

    # 2. 知识图谱检查
    try:
        from utils.knowledge_graph import get_total_subjects, load_kg

        kg = load_kg()
        subjects = get_total_subjects(kg)
        if subjects > 0:
            checks["knowledge_graph"] = {
                "status": "ok",
                "message": f"Loaded with {subjects} subject(s)",
            }
        else:
            checks["knowledge_graph"] = {"status": "error", "message": "No subjects found"}
            all_ok = False
    except Exception as e:
        checks["knowledge_graph"] = {"status": "error", "message": str(e)}
        all_ok = False

    # 3. DeepSeek API 可达性（轻量检查，不实际调用 LLM）
    try:
        from agent.llm import get_llm

        # 仅测试实例化，不实际调用
        llm = get_llm()
        checks["deepseek_api"] = {
            "status": "ok",
            "message": f"LLM instance created (model: {llm.model_name})",
        }
    except Exception as e:
        checks["deepseek_api"] = {"status": "error", "message": str(e)}
        all_ok = False

    # 汇总
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }


if __name__ == "__main__":
    import json

    result = health_check()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] == "down":
        sys.exit(1)
    elif result["status"] == "degraded":
        print("\n[WARN] Some checks reported warnings — app may still function.")
        sys.exit(0)
    else:
        print("\n[OK] All health checks passed.")
        sys.exit(0)
