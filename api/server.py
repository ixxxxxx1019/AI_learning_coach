"""
FastAPI 应用实例。

提供：
- REST API（/api/*）
- Swagger UI（/docs）
- 生命周期管理（startup/shutdown）
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.logging_config import get_logger, setup_logging
from config.settings import get_settings

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热，关闭时清理。"""
    logger.info("fastapi_startup", app=settings.app_name)
    # 预热：预加载 graph 和 KG
    from api.deps import get_graph, get_kg

    get_kg()
    get_graph()
    logger.info("fastapi_ready")
    yield
    logger.info("fastapi_shutdown")


app = FastAPI(
    title=f"{settings.app_name} API",
    description="AI 学习教练 REST API — 规划 → 教学 → 测验 → 批改 → 诊断",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — 允许跨域访问（开发环境宽松，生产环境应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)
