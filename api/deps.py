"""
FastAPI 依赖注入。

管理全局单例（graph、knowledge graph），
避免每次请求都重新创建。
"""

from agent.graph import create_graph
from config.logging_config import get_logger
from utils.knowledge_graph import load_kg

logger = get_logger(__name__)

# ---- 全局单例（模块级延迟初始化） ----
_graph = None
_kg = None


def get_graph():
    """获取 LangGraph 实例（单例）。

    LangGraph 的 checkpointer 需要跨请求保持状态，
    因此必须复用同一个 graph 实例。
    """
    global _graph
    if _graph is None:
        _graph = create_graph()
        logger.info("graph_instance_created")
    return _graph


def get_kg():
    """获取知识图谱（单例，内存缓存）。"""
    global _kg
    if _kg is None:
        _kg = load_kg()
        logger.info("kg_loaded")
    return _kg
