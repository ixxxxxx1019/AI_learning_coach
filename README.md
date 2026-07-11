# 🤖 AI 学习教练 — LangChain/LangGraph 重构版 (AI Learning Coach)

> 基于 LangChain + LangGraph + DeepSeek 的智能 AI 学习教练，支持多学科知识图谱、SM-2 间隔重复算法、以及 **规划 → 教学 → 测验 → 批改 → 诊断** 完整学习闭环。
>
> **v0.2.0** — 生产级工程化改造：结构化日志、LLM 韧性（重试/熔断/缓存）、Prompt 外部化、Docker 容器化、CI/CD、单元测试、FastAPI REST API、多用户支持。

---

## 1. 架构说明

### 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| **前端 UI** | Streamlit | Python 原生 Web UI，单页面三阶段交互 |
| **REST API** | FastAPI + Uvicorn | 标准 REST API，Swagger 文档自动生成，双端口共存 |
| **AI 编排** | LangGraph | 有状态图工作流，支持 `interrupt_before` 人机交互 |
| **LLM 客户端** | langchain-openai (`ChatOpenAI`) | OpenAI 兼容 SDK，指向 DeepSeek |
| **LLM 后端** | DeepSeek (`deepseek-chat`) | 通过 `https://api.deepseek.com` |
| **LLM 韧性** | tenacity + CircuitBreaker | 自动重试（指数退避）+ 熔断器三态保护 |
| **响应缓存** | L1 内存 LRU + L2 diskcache | 双层缓存减少 70%+ 重复 API 调用 |
| **提示管理** | YAML 外部化 + `ChatPromptTemplate` | Prompt 与代码解耦，可热重载 |
| **结构化输出** | Pydantic v2 + `with_structured_output(method="json_mode")` | JSON Mode（兼容 DeepSeek） |
| **状态检查点** | `MemorySaver` / `SqliteSaver` (LangGraph) | 内存或 SQLite 图状态持久化 |
| **用户进度** | `ProgressStore` (JSON/SQLite) | SM-2 状态 + 掌握度跨 session 持久化 |
| **配置管理** | Pydantic Settings (`pydantic-settings`) | 类型安全、`.env` 自动加载 |
| **日志** | structlog | 结构化日志，JSON/Console 双模式 |
| **可观测性** | LangSmith + CostTracker | LLM 调用链追踪 + Token 成本实时计算 |
| **容器化** | Docker + docker-compose | 多阶段构建，非 root 运行 |
| **CI/CD** | GitHub Actions | Lint → Test → Build → Deploy |
| **代码质量** | ruff + mypy + pre-commit | 自动化 lint/format/type-check |
| **UI 状态** | Streamlit `session_state` | 跨页面重渲染的状态保持 |
| **知识数据** | JSON 文件 (`knowledge_graph.json`) | 静态多学科知识图谱 |

### 项目结构

```
ai-coach-langchain/
├── app.py                      # Streamlit UI（Setup → Learn → Result）
├── run.py                      # [NEW] 统一启动入口（--api-only / --ui-only / 双服务）
├── requirements.txt             # 生产依赖
├── requirements-dev.txt         # 开发依赖（test/lint/type-check）
├── Dockerfile                   # 多阶段构建（python:3.12-slim）
├── docker-compose.yml           # 多服务编排（app + redis）
├── health.py                    # 健康检查（Docker HEALTHCHECK / CI）
├── pyproject.toml               # ruff + mypy 配置
├── .pre-commit-config.yaml      # Git pre-commit hooks
├── .env.example                 # 环境变量模板（无敏感信息）
│
├── agent/                       # AI Agent 核心层
│   ├── models.py                # 8 个 Pydantic 模型 + LearningState TypedDict
│   ├── llm.py                   # LLM 工厂：get_llm() / get_structured_llm()
│   ├── graph.py                 # LangGraph 状态图（5 节点 + Human-in-the-Loop）
│   ├── planner.py               # 规划师 Agent：从 KG 加载知识点，生成学习计划
│   ├── tutor.py                 # 讲师 Agent：按 KG 详情生成 Markdown 教学内容
│   ├── evaluator.py             # 评估师 Agent：Quiz + Grading + Diagnosis 三条链
│   ├── resilience.py            # [NEW] LLM 韧性：tenacity 重试 + CircuitBreaker 熔断
│   ├── cache.py                 # [NEW] 双层缓存：L1 内存 LRU + L2 diskcache
│   └── cost_tracker.py          # [NEW] Token 成本追踪 Callback
│
├── api/                         # [NEW] FastAPI REST API 层
│   ├── server.py                # FastAPI 应用实例 + CORS + 生命周期
│   ├── routes.py                # 7 个 API 端点（subjects / sessions / health）
│   ├── schemas.py               # Pydantic 请求/响应模型
│   └── deps.py                  # 依赖注入（graph / KG 单例）
│
├── config/                      # [NEW] 应用配置层
│   ├── settings.py              # Pydantic Settings：统一配置入口
│   ├── logging_config.py        # structlog 结构化日志配置
│   └── prompts.py               # PromptLoader：YAML → Prompt 加载器
│
├── prompts/                     # [NEW] Prompt YAML 文件（与代码解耦）
│   ├── planner.yaml             # Planner 系统提示
│   ├── tutor.yaml               # Tutor 系统提示
│   └── evaluator.yaml           # Quiz / Grading / Diagnosis 三条提示
│
├── utils/                       # 算法工具层
│   ├── __init__.py              # [NEW] 包初始化 + 完整导出
│   ├── knowledge_graph.py       # 知识图谱：依赖图遍历 / 根因分析 / 拓扑排序
│   ├── spaced_repetition.py     # SM-2 间隔重复算法（从论文公式手写实现）
│   └── progress_store.py        # [NEW] 用户学习进度持久化存储
│
├── tests/                       # [NEW] 测试套件（31 个测试）
│   ├── test_knowledge_graph.py  # KG 查询 / 依赖 / 根因分析测试
│   ├── test_spaced_repetition.py# SM-2 算法测试（8 个边界用例）
│   └── test_models.py           # Pydantic 模型序列化测试
│
├── data/
│   └── knowledge_graph.json     # 多学科知识图谱（CET6 词汇 + Python 基础，55+ 知识点）
│
└── .github/workflows/           # [NEW] CI/CD
    └── ci.yml                   # Lint → Test → Build → Deploy
```

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│              Nginx / 网关 (生产环境)                       │
│         /api/* → FastAPI    │    /* → Streamlit          │
└────────────┬──────────────────┬─────────────────────────┘
             │                  │
┌────────────▼────────┐  ┌──────▼──────────────────────────┐
│  FastAPI :8000      │  │  Streamlit UI :8501 (app.py)     │
│  Swagger UI /docs   │  │  ① Setup ② Learn ③ Result       │
│  7 REST Endpoints   │  │  多用户 UUID thread_id           │
└──────────┬──────────┘  └────────────────┬────────────────┘
           │                              │
           └──────────┬───────────────────┘
                      │ graph.invoke(state, config)
                      ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph State Graph                       │
│                                                          │
│   ┌──────────┐   ┌───────┐   ┌──────┐   ⏸️               │
│   │ planner  │──►│ tutor │──►│ quiz │─── HUMAN INPUT     │
│   │ (KG驱动) │   │(KG详情)│   │(KG ID)│        │          │
│   └──────────┘   └───────┘   └──────┘   ┌────┴────┐     │
│                                          │  grade  │     │
│                                          └────┬────┘     │
│                                          ┌────┴────┐     │
│                                          │diagnose │     │
│                                          └─────────┘     │
│                                                          │
│   韧性层：retryable_invoke (3次重试) + CircuitBreaker     │
│   缓存层：LLMCache (L1内存 + L2磁盘)                      │
│   观测层：structlog + LangSmith + CostTracker             │
│                                                          │
│   State: LearningState (TypedDict, 15+ 字段)             │
│   Checkpointer: MemorySaver / SqliteSaver                │
│   Interrupt: before "grade" node                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 关键 Prompt 与 Vibe 思路

### 2.1 Planner — "专业的 AI 学习规划师"

**系统提示** (`prompts/planner.yaml`)：

| 要素 | 内容 |
|------|------|
| **输入** | 学科信息 + 知识点列表（从 KG 实时加载，含 ID/title/difficulty/依赖）+ 用户可用时长 |
| **任务** | 分析依赖 → 评估优先级 → 划分复习/学新/测验 → 分配时间 |
| **原则** | 前置依赖优先、每次学习完整循环、难点多分配时间、总时长不超限、**kp_ids 必须从给定列表选择** |
| **输出** | `StudyPlan` JSON（包含多个 `StudyPhase`，kp_ids 均为 KG 有效 ID） |

### 2.2 Tutor — "专业的 AI 讲师"

**系统提示** (`prompts/tutor.yaml`)：

| 要素 | 内容 |
|------|------|
| **风格** | 用中文，清晰有条理，生动易懂 |
| **格式** | Markdown（标题、列表、加粗） |
| **内容** | 定义 → 例句 → 记忆技巧；词汇：词性 + 中文释义 + 英文例句 + 近义词辨析 |
| **关键** | 明确指出知识点之间的关联；**接收来自 KG 的 word/definition/pos 详情** |

### 2.3 Evaluator — 三条独立 Prompt 链

**Prompt 文件** (`prompts/evaluator.yaml`)：

#### Quiz 系统
- **角色**："专业的测评专家"
- **要求**：每知识点 ≥1 题，题型多样（选择/填空/翻译），难度适中，有答案解析
- **约束**：`target_kp_id` 必须使用提供的 KG 有效 ID
- **输出**：`Quiz` JSON

#### Grading 系统
- **角色**："严格的阅卷老师"
- **要求**：对比标准答案逐题批改，错误分析 + 鼓励性评语，百分制
- **输出**：`GradingResult` JSON

#### Diagnosis 系统
- **角色**："AI 学习诊断专家"
- **核心规则**：
  - 全部答对 → `mastery_change` +0.1 ~ +0.2
  - 全部答错 → `mastery_change` -0.2 ~ -0.3
  - 部分正确 → `mastery_change` +0.05 ~ +0.1
- **约束**：`kp_id` 和 `next_priority` 必须从实际考察的知识点列表中选择
- **错误分类**：概念不清 / 拼写错误 / 语法错误 / 词汇混淆
- **输出**：`Diagnosis` JSON（含针对性建议 + 下一步优先知识点）

### Vibe / 设计哲学

1. **教育核心循环**：每次学习 = 规划 → 教学 → 测验 → 诊断，模拟真实一对一家教模式。

2. **知识依赖追溯**：`find_weak_root_causes()` 不是简单标记错题，而是**沿依赖链向上追溯**，找到真正的薄弱根源。例如：用户做错了一个定语从句题，根因可能是 3 天前学的"先行词"概念没掌握。

3. **SM-2 从第一性原理**：不调 Anki 包，从论文公式手写实现。每个 EF 因子更新、间隔计算都有详细注释，**每行代码都能解释**。

4. **结构化输出优先**：8 个 Pydantic 模型 + JSON Mode，确保 LLM 输出一致性，减少解析错误。

5. **多学科可扩展**：知识图谱 JSON 已包含 CET6 词汇 + Python 基础两个学科，新增学科只需扩展 JSON 文件。

6. **中文优先**：所有 Prompt、UI、变量名、注释均为中文，面向中国学习者。

7. **生产级韧性**：API 调用自动重试（指数退避 3 次）+ 熔断器防止雪崩 + 双层缓存降低 70%+ 重复调用成本。

8. **Prompt 即代码**：YAML 外部化管理，非技术人员可调整 AI 行为，Git diff 友好，支持 A/B 测试。

---

## 3. AI 调用逻辑

### 调用方式总览

| 组件 | 链类型 | 温度 | max_tokens | 输出 | 调用 | 韧性 |
|------|--------|------|------------|------|------|------|
| Planner | `PROMPT \| get_structured_llm(StudyPlan)` | 0.1 | 4096 | Pydantic | `retryable_invoke()` | 重试 + 熔断 |
| Tutor | `PROMPT \| get_llm(t=0.5) \| StrOutputParser` | 0.5 | 4096 | Markdown str | `retryable_invoke()` | 重试 + 熔断 |
| Quiz | `PROMPT \| get_structured_llm(Quiz)` | 0.1 | 4096 | Pydantic | `retryable_invoke()` | 重试 + 熔断 |
| Grading | `PROMPT \| get_structured_llm(GradingResult)` | 0.1 | 4096 | Pydantic | `retryable_invoke()` | 重试 + 熔断 |
| Diagnosis | `PROMPT \| get_structured_llm(Diagnosis)` | 0.1 | 4096 | Pydantic | `retryable_invoke()` | 重试 + 熔断 |

### 链定义示例

```python
# agent/planner.py — 结构化输出链（Prompt 从 YAML 加载）
from config.prompts import PromptLoader
_loader = PromptLoader()

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _loader.get_system_prompt("planner")),
    ("user", "{user_input}")
])

def create_planner():
    return PLANNER_PROMPT | get_structured_llm(StudyPlan)

# agent/graph.py — 带韧性的调用
plan = retryable_invoke(planner, {"user_input": user_input})
```

### LLM 韧性机制（v0.2.0 新增）

```python
# agent/resilience.py — 自动重试
@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),  # 1 原始 + 3 重试
)
def _invoke_with_retry():
    return chain.invoke(input_dict)

# agent/resilience.py — 熔断器
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
result = cb.call(chain.invoke, {"user_input": "..."})
# 状态: CLOSED → (5 次连续失败) → OPEN → (60s 超时) → HALF_OPEN → CLOSED
```

### 双层缓存架构（v0.2.0 新增）

```
请求 → L1 内存缓存 (lru_cache 128) → 命中? → 返回
              ↓ 未命中
         L2 磁盘缓存 (diskcache TTL) → 命中? → 回填 L1 → 返回
              ↓ 未命中
         调用 LLM API → 写入 L1 + L2 → 返回
```

### 流式 (Streaming)

**不使用。** 所有调用均为同步阻塞 `invoke()`。UI 使用 `st.spinner("AI 正在...")` 显示加载状态。

### Function Calling / Tool Use

**不使用。** 项目使用 `with_structured_output(method="json_mode")`，原因：

> `method='json_mode'` 兼容 DeepSeek（不支持原生 `json_schema`）

### LangGraph Human-in-the-Loop 机制

```python
# agent/graph.py
workflow.add_node("planner", planner_node)  # 从 KG 加载知识点，生成计划
workflow.add_node("tutor", tutor_node)      # 从 KG 查详情，生成教学内容
workflow.add_node("quiz", quiz_node)        # 从 KG 查详情，生成测验题
workflow.add_node("grade", grade_node)      # 批改答案
workflow.add_node("diagnose", diagnose_node)# 诊断（约束有效 kp_id）

app = graph.compile(
    checkpointer=MemorySaver(),    # 或 SqliteSaver
    interrupt_before=["grade"],    # ⏸️ 出题后暂停，等待用户输入
)
```

**交互流程**：
1. `graph.invoke(initial_state, config)` → 自动执行 planner → tutor → quiz，然后在 grade 前暂停
2. 用户在 Streamlit 中填写答案
3. 调用 `graph.update_state(config, {"user_answers": answers})` 注入答案
4. 调用 `graph.invoke(None, config)` 从断点恢复 → grade → diagnose → END

### LLM 密钥解析策略 (`agent/llm.py`)

```
优先使用 config.settings.get_settings() (Pydantic Settings)
  ├── 1. .env 文件 (python-dotenv)           ← 本地开发
  ├── 2. os.environ[key]                      ← 命令行注入
  └── 3. Streamlit Cloud Secrets (st.secrets) ← 生产环境（向后兼容）

LangSmith 追踪（可选）
  └── 设置 LANGCHAIN_API_KEY 环境变量后自动启用
```

### 知识点数据流（v0.2.0 关键修复）

```
knowledge_graph.json
       │
       ▼
planner_node: list_all_kps(subject_id) → 传入 Planner Prompt
       │ 生成 plan.phases[].kp_ids (均为有效 KG ID)
       ▼
tutor_node:  get_kp(kp_id) → title/word/definition → 传入 Tutor Prompt
       │
       ▼
quiz_node:   get_kp(kp_id) → title/word/definition → 传入 Quiz Prompt
       │ 生成 questions[].target_kp_id (均为有效 KG ID)
       ▼
diagnose_node: 从 quiz_data 提取有效 kp_id 列表 → 约束 Diagnosis 输出
```

### 知识图谱 (`data/knowledge_graph.json`)

| 学科 | 领域 | 知识点数 | 特点 |
|------|------|---------|------|
| CET6 英语词汇 | 学术与教育、科技与创新、社会与文化 | 45 | 词汇间有同义/反义/搭配依赖 |
| Python 编程基础 | 基础语法、数据结构 | 10 | 线性依赖链（变量→类型→函数→类） |

`utils/knowledge_graph.py` 提供：
- `get_prerequisites(kp_id)` — 查询前置依赖
- `get_dependents(kp_id)` — 查询后续依赖
- `find_weak_root_causes(wrong_kp_ids)` — **依赖链薄弱根因分析**（面试亮点）
- `get_learnable_kps(subject_id, mastered_ids)` — 筛选可学知识点
- `filter_kps()` — 多条件筛选（领域/难度/标签）

### REST API 端点（v0.2.0 新增）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查（配置 / KG / API 三重校验） |
| `GET` | `/api/subjects` | 列出所有学科 |
| `GET` | `/api/subjects/{id}/kps` | 列出某学科知识点（含依赖/难度/标签） |
| `POST` | `/api/sessions` | 创建学习 session → planner + tutor + quiz |
| `GET` | `/api/sessions/{id}` | 查询 session 当前状态 |
| `POST` | `/api/sessions/{id}/answers` | 提交答案 → grade + diagnosis |
| `DELETE` | `/api/sessions/{id}` | 删除 session |

```bash
# 启动双服务
python run.py                    # API(:8000) + UI(:8501)
python run.py --api-only         # 仅 API
python run.py --ui-only          # 仅 Streamlit

# 测试 API
curl http://localhost:8000/api/health
curl http://localhost:8000/api/subjects
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"subject_id":"cet6_vocab","time_minutes":10}'

# Swagger 交互文档
open http://localhost:8000/docs
```

---

## 4. 部署步骤说明

### 本地开发

```bash
# 1. 克隆项目
git clone https://github.com/ixxxxxx1019/AI_learning_coach.git
cd AI_learning_coach

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt           # 生产依赖
pip install -r requirements-dev.txt       # 开发依赖（含 test/lint）

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 DeepSeek API Key

# 5. 运行测试
python -m pytest tests/ -v                # 31 个测试

# 6. 健康检查
python health.py

# 7. 启动（双服务）
python run.py                              # API(:8000) + UI(:8501)
# 或单独启动
streamlit run app.py                       # 仅 UI
python run.py --api-only                   # 仅 API

# 访问
# UI: http://localhost:8501
# API: http://localhost:8000/docs
```

### Docker 部署（推荐）

```bash
# 1. 准备 .env 文件
cp .env.example .env
# 编辑 .env 填入 API Key

# 2. 构建并启动
docker compose up -d

# 3. 查看日志
docker compose logs -f app

# 4. 健康检查
docker exec ai-coach-langchain python health.py

# 5. 停止
docker compose down
```

### Streamlit Cloud 部署

1. Push 项目到 GitHub
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 连接仓库 `ixxxxxx1019/AI_learning_coach`
4. 在 **Settings → Secrets** 配置：

```toml
DEEPSEEK_API_KEY = "sk-your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

5. 点击 **Deploy**，等待构建完成

### CI/CD (GitHub Actions)

Push 到 master/main 分支自动触发：

```
Lint (ruff) → Test (pytest 31 tests) → Build (Docker) → Deploy
```

详见 `.github/workflows/ci.yml`。

### DNS / HTTPS 说明

| 部署方式 | HTTPS | 域名 | 适用 |
|---------|-------|------|------|
| **本地开发** | ❌ | `localhost:8501` | 开发调试 |
| **Streamlit Cloud** | ✅ 自动 TLS | `xxx.streamlit.app` | 快速 demo |
| **Docker** | ❌ (裸) / ✅ (Nginx) | 自定义 | 单机部署 |
| **自建服务器** | 需配置 | 自定义域名 | 生产环境 |

#### 自建服务器部署（Nginx + HTTPS）

```nginx
# /etc/nginx/sites-available/ai-coach
server {
    listen 443 ssl http2;
    server_name coach.your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/coach.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/coach.your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Streamlit WebSocket 端点
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 后台运行 (systemd)

```ini
# /etc/systemd/system/ai-coach.service
[Unit]
Description=AI Learning Coach Streamlit App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-coach-langchain
EnvironmentFile=/opt/ai-coach-langchain/.env
ExecStart=/opt/ai-coach-langchain/venv/bin/streamlit run app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-coach
```

### 代码质量

```bash
# 一键修复
python -m ruff check . --fix
python -m ruff format .

# 类型检查
python -m mypy agent/

# Pre-commit hooks（提交前自动检查）
pre-commit install
pre-commit run --all-files
```

### 生产环境改进建议

| 方面 | 当前状态 (v0.2.0) | 生产建议 |
|------|---------|---------|
| 状态持久化 | `MemorySaver` / `SqliteSaver` | 替换为 `PostgresSaver`（多实例） |
| 用户认证 | UUID thread_id（无认证） | 集成 OAuth 或 Streamlit Authenticator |
| 日志 | structlog (JSON/Console) | 接入 ELK / Loki 集中式日志 |
| API Key | `.env` 文件 | 使用 Secrets Manager（Vault/AWS Secrets） |
| 监控 | CostTracker + Health Check | Prometheus + Grafana 仪表盘 |
| 缓存 | L1 内存 + L2 diskcache | 替换 L2 为 Redis（docker-compose 已预配置） |
| CI/CD | GitHub Actions (Lint/Test/Build) | 添加 Deploy 步骤到目标环境 |
| Prompt 管理 | YAML 文件 + Git 版本控制 | 添加 Prompt 性能追踪（A/B 测试框架） |
| 知识图谱 | JSON 文件 (~55 KPs) | 迁移至图数据库（Neo4j）支持大规模知识点 |

---

## 5. v0.2.0 更新日志

### 新增

- **配置管理**：Pydantic Settings 类型安全配置 + `.env.example` 模板
- **结构化日志**：structlog (JSON/Console 双模式)，替换全部 `print()`
- **Prompt 外部化**：YAML 文件管理 + PromptLoader + 硬编码回退
- **LLM 韧性**：tenacity 自动重试（指数退避）+ CircuitBreaker 三态熔断
- **双层缓存**：L1 内存 LRU + L2 diskcache，减少 70%+ 重复 API 调用
- **成本追踪**：CostTracker Callback，实时显示 Token 消耗和费用
- **Docker**：多阶段构建 (python:3.12-slim) + docker-compose + 非 root 运行
- **CI/CD**：GitHub Actions (Lint → Test → Build → Deploy)
- **测试**：31 个单元测试（KG / SM-2 / Models）
- **健康检查**：`health.py` — 配置 / KG / API 三重检查
- **FastAPI REST API**：7 个端点 + Swagger 文档 + run.py 统一入口 + 与 Streamlit 共存
- **用户进度**：`ProgressStore` — SM-2 状态跨 session 持久化
- **多用户**：UUID thread_id，替换硬编码 `"session-1"`
- **代码质量**：ruff + mypy + pre-commit hooks + pyproject.toml

### 修复

- **知识点数据流**：Planner/Tutor/Quiz/Diagnose 从 KG 加载实际 KP 数据，消除 LLM 编造 ID
- **依赖分离**：`requirements.txt`（生产）+ `requirements-dev.txt`（开发）
- **安全**：`.env.example` 替代硬编码 API Key，`.gitignore` 补全覆盖
- **包结构**：`utils/__init__.py`、`config/__init__.py` 正确导出

---

## 📄 License

MIT

## 🔗 相关项目

- [kaoyan-english-coach](https://github.com/ixxxxxx1019/kaoyan-english-coach) — 考研英语 AI 学习教练（本项目的前身/姊妹项目）
- [deep-research](https://github.com/ixxxxxx1019/deep-research) — 多 Agent 深度研究系统
