# 🤖 AI 学习教练 — LangChain/LangGraph 重构版 (AI Learning Coach)

> 基于 LangChain + LangGraph + DeepSeek 的智能 AI 学习教练，支持多学科知识图谱、SM-2 间隔重复算法、以及 **规划 → 教学 → 测验 → 批改 → 诊断** 完整学习闭环。

---

## 1. 架构说明

### 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| **前端 UI** | Streamlit | Python 原生 Web UI，单页面三阶段交互 |
| **AI 编排** | LangGraph | 有状态图工作流，支持 `interrupt_before` 人机交互 |
| **LLM 客户端** | langchain-openai (`ChatOpenAI`) | OpenAI 兼容 SDK，指向 DeepSeek |
| **LLM 后端** | DeepSeek (`deepseek-chat`) | 通过 `https://api.deepseek.com` |
| **提示管理** | `langchain_core.prompts.ChatPromptTemplate` | System/User 消息模板 |
| **结构化输出** | Pydantic v2 + `with_structured_output(method="json_mode")` | JSON Mode（兼容 DeepSeek） |
| **状态检查点** | `MemorySaver` (LangGraph) | 内存级图状态持久化 |
| **UI 状态** | Streamlit `session_state` | 跨页面重渲染的状态保持 |
| **知识数据** | JSON 文件 (`knowledge_graph.json`) | 静态多学科知识图谱 |

### 项目结构

```
ai-coach-langchain/
├── app.py                    # 主入口：Streamlit UI（Setup → Learn → Result）
├── requirements.txt          # Python 依赖
│
├── agent/                    # AI Agent 核心层
│   ├── models.py             # 8 个 Pydantic 模型 + LearningState TypedDict
│   ├── llm.py                # LLM 工厂：get_llm() / get_structured_llm()
│   ├── graph.py              # LangGraph 状态图（5 节点 + Human-in-the-Loop）
│   ├── planner.py            # 规划师 Agent：生成个性化学习计划
│   ├── tutor.py              # 讲师 Agent：生成 Markdown 教学内容
│   └── evaluator.py          # 评估师 Agent：Quiz + Grading + Diagnosis 三条链
│
├── utils/                    # 算法工具层
│   ├── knowledge_graph.py    # 知识图谱：依赖图遍历 / 根因分析 / 拓扑排序
│   └── spaced_repetition.py  # SM-2 间隔重复算法（从论文公式手写实现）
│
├── data/
│   └── knowledge_graph.json  # 多学科知识图谱（CET6 词汇 + Python 基础）
│
└── scripts/                  # 部署脚本目录（预留）
```

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)                    │
│  ① Setup: 选学科 + 设时长  │  ② Learn: 内容+答题  │  ③ Result │
└──────────────────────┬──────────────────────────────────┘
                       │ graph.invoke(initial_state, config)
                       ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph State Graph                       │
│                                                          │
│   ┌──────────┐   ┌───────┐   ┌──────┐   ⏸️               │
│   │ planner  │──►│ tutor │──►│ quiz │─── HUMAN INPUT     │
│   │ 生成计划 │   │ 教学  │   │ 出题 │    (用户答题)      │
│   └──────────┘   └───────┘   └──────┘        │          │
│                                          ┌────┴────┐     │
│                                          │  grade  │     │
│                                          │  批改   │     │
│                                          └────┬────┘     │
│                                          ┌────┴────┐     │
│                                          │diagnose │     │
│                                          │  诊断   │     │
│                                          └─────────┘     │
│                                                          │
│   State: LearningState (TypedDict, 15+ 字段)             │
│   Checkpointer: MemorySaver                              │
│   Interrupt: before "grade" node                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 关键 Prompt 与 Vibe 思路

### 2.1 Planner — "专业的 AI 学习规划师"

**系统提示** (`agent/planner.py`)：

| 要素 | 内容 |
|------|------|
| **输入** | 学科信息 + 知识点列表（含依赖关系） + 用户可用时长 |
| **任务** | 分析依赖 → 评估优先级 → 划分复习/学新/测验 → 分配时间 |
| **原则** | 前置依赖优先、每次学习完整循环、难点多分配时间、总时长不超限 |
| **输出** | `StudyPlan` JSON（包含多个 `StudyPhase`） |

### 2.2 Tutor — "专业的 AI 讲师"

**系统提示** (`agent/tutor.py`)：

| 要素 | 内容 |
|------|------|
| **风格** | 用中文，清晰有条理，生动易懂 |
| **格式** | Markdown（标题、列表、加粗） |
| **内容** | 定义 → 例句 → 记忆技巧；词汇：词性 + 中文释义 + 英文例句 + 近义词辨析 |
| **关键** | 明确指出知识点之间的关联 |

### 2.3 Evaluator — 三条独立 Prompt 链

#### Quiz 系统
- **角色**："专业的测评专家"
- **要求**：每知识点 ≥1 题，题型多样（选择/填空/翻译），难度适中，有答案解析
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
- **错误分类**：概念不清 / 拼写错误 / 语法错误 / 词汇混淆
- **输出**：`Diagnosis` JSON（含针对性建议 + 下一步优先知识点）

### Vibe / 设计哲学

1. **教育核心循环**：每次学习 = 规划 → 教学 → 测验 → 诊断，模拟真实一对一家教模式。

2. **知识依赖追溯**：`find_weak_root_causes()` 不是简单标记错题，而是**沿依赖链向上追溯**，找到真正的薄弱根源。例如：用户做错了一个定语从句题，根因可能是 3 天前学的"先行词"概念没掌握。

3. **SM-2 从第一性原理**：不调 Anki 包，从论文公式手写实现。每个 EF 因子更新、间隔计算都有详细注释，**每行代码都能解释**。

4. **结构化输出优先**：8 个 Pydantic 模型 + JSON Mode，确保 LLM 输出一致性，减少解析错误。

5. **多学科可扩展**：知识图谱 JSON 已包含 CET6 词汇 + Python 基础两个学科，新增学科只需扩展 JSON 文件。

6. **中文优先**：所有 Prompt、UI、变量名、注释均为中文，面向中国学习者。

---

## 3. AI 调用逻辑

### 调用方式总览

| 组件 | 链类型 | 温度 | max_tokens | 输出 | 调用 |
|------|--------|------|------------|------|------|
| Planner | `PROMPT \| get_structured_llm(StudyPlan)` | 0.1 | 4096 | Pydantic | `.invoke()` |
| Tutor | `PROMPT \| get_llm(t=0.5) \| StrOutputParser` | 0.5 | 4096 | Markdown str | `.invoke()` |
| Quiz | `PROMPT \| get_structured_llm(Quiz)` | 0.1 | 4096 | Pydantic | `.invoke()` |
| Grading | `PROMPT \| get_structured_llm(GradingResult)` | 0.1 | 4096 | Pydantic | `.invoke()` |
| Diagnosis | `PROMPT \| get_structured_llm(Diagnosis)` | 0.1 | 4096 | Pydantic | `.invoke()` |

### 链定义示例

```python
# agent/planner.py — 结构化输出链
chain = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    ("user", "{user_input}")
]) | get_structured_llm(StudyPlan)

result: StudyPlan = chain.invoke({"user_input": "..."})

# agent/tutor.py — 文本输出链
chain = ChatPromptTemplate.from_messages([
    ("system", TUTOR_SYSTEM_PROMPT),
    ("user", "{user_input}")
]) | get_llm(temperature=0.5) | StrOutputParser()

result: str = chain.invoke({"user_input": "..."})
```

### 流式 (Streaming)

**不使用。** 所有调用均为同步阻塞 `invoke()`。UI 使用 `st.spinner("AI 正在...")` 显示加载状态。

### Function Calling / Tool Use

**不使用。** 项目使用 `with_structured_output(method="json_mode")`，原因：

> `method='json_mode'` 兼容 DeepSeek（不支持原生 `json_schema`）

DeepSeek API 不支持 OpenAI 的 `response_format: { type: "json_schema" }` 原生 JSON Schema 模式，因此使用 `json_mode`（即 `response_format: { type: "json_object" }`）作为替代。

### LangGraph Human-in-the-Loop 机制

```python
# agent/graph.py
graph = StateGraph(LearningState)
graph.add_node("planner", planner_node)
graph.add_node("tutor", tutor_node)
graph.add_node("quiz", quiz_node)
graph.add_node("grade", grade_node)
graph.add_node("diagnose", diagnose_node)

# ... edges ...

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["grade"]  # ⏸️ 出题后暂停，等待用户输入
)
```

**交互流程**：
1. `graph.invoke(initial_state, config)` → 自动执行 planner → tutor → quiz，然后在 grade 前暂停
2. 用户在 Streamlit 中填写答案
3. 调用 `graph.update_state(config, {"user_answers": answers})` 注入答案
4. 调用 `graph.invoke(None, config)` 从断点恢复 → grade → diagnose → END

### LLM 密钥解析策略 (`agent/llm.py`)

```
_get_secret(key)
  ├── 1. Streamlit Cloud Secrets (st.secrets)  ← 生产环境优先
  ├── 2. os.environ[key]                       ← 命令行注入
  └── 3. .env 文件 (python-dotenv)             ← 本地开发
```

### 知识图谱 (`data/knowledge_graph.json`)

| 学科 | 领域 | 知识点数 | 特点 |
|------|------|---------|------|
| CET6 英语词汇 | 学术与教育、科技与创新、社会与文化 | 30+ | 词汇间有同义/反义/搭配依赖 |
| Python 编程基础 | 基础语法、数据结构 | 10+ | 线性依赖链（变量→类型→函数→类） |

`utils/knowledge_graph.py` 提供：
- `get_prerequisites(kp_id)` — 查询前置依赖
- `get_dependents(kp_id)` — 查询后续依赖
- `find_weak_root_causes(wrong_kp_ids)` — **依赖链薄弱根因分析**（面试亮点）
- `topological_sort()` — 拓扑排序

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
pip install -r requirements.txt

# 4. 配置环境变量
# 创建 .env 文件，填入：
#   DEEPSEEK_API_KEY=sk-your-key
#   DEEPSEEK_BASE_URL=https://api.deepseek.com

# 5. 启动
streamlit run app.py
# 访问 http://localhost:8501
```

### Streamlit Cloud 部署（推荐）

1. Push 项目到 GitHub
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 连接仓库 `ixxxxxx1019/AI_learning_coach`
4. 在 **Settings → Secrets** 配置：

```toml
DEEPSEEK_API_KEY = "sk-your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

5. 点击 **Deploy**，等待构建完成

### DNS / HTTPS 说明

| 部署方式 | HTTPS | 域名 | 适用 |
|---------|-------|------|------|
| **本地开发** | ❌ | `localhost:8501` | 开发调试 |
| **Streamlit Cloud** | ✅ 自动 TLS | `xxx.streamlit.app` | 快速 demo |
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

获取免费 TLS 证书：

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d coach.your-domain.com
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
Environment="DEEPSEEK_API_KEY=sk-xxx"
Environment="DEEPSEEK_BASE_URL=https://api.deepseek.com"
ExecStart=/opt/ai-coach-langchain/venv/bin/streamlit run app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-coach
```

### 生产环境改进建议

| 方面 | 当前状态 | 生产建议 |
|------|---------|---------|
| 状态持久化 | `MemorySaver`（内存） | 替换为 `SqliteSaver` 或 `PostgresSaver` |
| 用户认证 | 无 | 集成 OAuth 或 Streamlit Authenticator |
| 日志 | `print()` | 使用 `logging` 模块 + 集中式日志 |
| API Key | `.env` 文件 | 使用 Secrets Manager 或环境变量注入 |
| 监控 | 无 | 添加 Prometheus + Grafana 或 Streamlit Cloud 自带监控 |

---

## 📄 License

MIT

## 🔗 相关项目

- [kaoyan-english-coach](https://github.com/ixxxxxx1019/kaoyan-english-coach) — 考研英语 AI 学习教练（本项目的前身/姊妹项目）
- [deep-research](https://github.com/ixxxxxx1019/deep-research) — 多 Agent 深度研究系统
