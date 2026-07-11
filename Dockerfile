# ============================================================
# AI学习教练 — 生产级 Dockerfile
# ============================================================
# 多阶段构建：builder 安装依赖，runtime 最小化镜像
#
# 构建：docker build -t ai-coach-langchain .
# 运行：docker run -p 8501:8501 --env-file .env ai-coach-langchain
# ============================================================

# ---- Builder Stage ----
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# 安装构建依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Runtime Stage ----
FROM python:3.12-slim-bookworm

WORKDIR /app

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 从 builder 复制已安装的包
COPY --from=builder /root/.local /home/appuser/.local

# 复制应用代码
COPY --chown=appuser:appuser . .

# 创建数据目录
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app/data /app/logs

# 设置环境变量
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python health.py || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
