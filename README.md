# StudyBuddy — AI 学习助手

面向初中生的全栈 AI 学习平台。拍照上传作业 → AI 批改 → 自动归档错题 → 知识点追踪 → 个性化复习。

接入 **EverMemOS** 长期记忆云服务，让 AI 家教真正跨会话"记住"每个学生——批改历史、薄弱知识点、学习偏好，越用越懂你。

---

## 演示视频

<video src="https://github.com/user-attachments/assets/eb99171f-246f-4a90-99e0-91e803813c4b" controls width="100%"></video>

---

## 功能概览

- **作业批改**：拍照上传，AI 自动识别题目、判断对错、给出解析和错误类型
- **知识点标注**：每道题自动标注涉及的知识点和难度等级，命中历史薄弱点即高亮提示
- **错题本**：错题自动归档，支持按科目/知识点筛选，AI 一键解析，闪卡复习
- **AI 问答**：多轮对话问难题，AI 会结合你的历史学情给出个性化讲解
- **学情追踪**：自动统计正确率、错误类型分布、各科薄弱知识点趋势

---

## 部署指南

### 环境要求

- Python 3.10+
- 任意支持视觉输入的 LLM API（推荐 Gemini 2.5 Pro、GPT-4o）
- （可选）EverMemOS API Key，用于跨会话长期记忆

### 第一步：克隆项目

```bash
git clone https://github.com/your-username/StudyBuddy.git
cd StudyBuddy
```

### 第二步：安装依赖

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 第三步：配置环境变量

复制模板文件，然后填入你的 API Key：

```bash
cp .env.example .env
```

打开 `.env`，至少填写以下必填项：

```bash
# 主 LLM（必须支持图片输入，用于 OCR 识别和批改）
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
LLM_HOST=https://api.openai.com/v1

# EverMemOS 长期记忆（可选，不填则跳过记忆功能）
EVERMEMOS_API_KEY=your-api-key
EVERMEMOS_USER_ID=my_studybuddy
```

完整配置项说明见 [`.env.example`](.env.example)。

> **推荐模型**：Gemini 2.5 Pro（性价比高，视觉能力强，支持 OpenAI 兼容接口）。
> 配置方式：`LLM_BINDING=openai`，`LLM_HOST=https://generativelanguage.googleapis.com/v1beta/openai/`。

### 第四步：启动服务

```bash
python src/api/run_server.py
```

默认监听 `http://0.0.0.0:8001`，端口可在 `.env` 中通过 `BACKEND_PORT` 修改。

### 第五步：打开应用

浏览器访问：

```
http://localhost:8001/ui/
```

**手机访问（局域网）**：确保手机和电脑在同一 Wi-Fi，用电脑的局域网 IP 访问：

```
http://192.168.x.x:8001/ui/
```

查看你的局域网 IP：

```bash
# macOS / Linux
ip addr show | grep "inet " | grep -v 127.0.0.1

# macOS 快捷方式
ipconfig getifaddr en0
```

---

## EverMemOS：让 AI 真正记住每一个学生

### 为什么需要长期记忆？

普通 AI 没有记忆，每次对话都从零开始。使用 StudyBuddy 一个月后，AI 依然不知道这个学生数学上的"因式分解"一直是薄弱点，也不记得上次讲解时学生在哪里卡住了。

[EverMemOS](https://evermind.ai) 解决这个问题——它是专为 AI 应用设计的云端长期记忆服务。StudyBuddy 每次批改后自动把学情写入 EverMemOS，下次 AI 开口前先检索相关记忆，再给出真正个性化的讲解。

### 写入了哪些记忆？

| 触发时机 | 写入的内容 |
|---------|-----------|
| 每次作业批改完成 | 科目、题数、正确率、错误类型分布 |
| 每道错题 | 题目原文、学生作答、正确答案、错误类型 |
| 检测到薄弱知识点 | 薄弱点列表，写为"复习提醒"记忆 |
| AI 对话达到轮次上限 | 本次会话摘要（学生问了什么 + AI 分析结论） |
| 修改个人设置 | 昵称、年级、MBTI、讲解风格偏好 |

### 记忆在哪里发挥作用？

**批改时（KnowPointAgent）**：注入学生历史薄弱知识点，自动标记哪些题命中了已知盲区（`is_weak_area=true`）。

**问答时（ExplainAgent）**：开口前先检索该学生的学情记忆，了解讲解风格偏好、近期薄弱点、常犯错误类型，再用个性化方式解答。

### 记忆类型说明

EverMemOS 将记忆分为三类，StudyBuddy 全部用到：

- **EventLog（事实型）**：成绩数据、错题明细——"学生上周数学正确率 72%，错了 3 道因式分解"
- **Foresight（预见型）**：薄弱点复习提醒——"下次遇到平方差公式时，注意上次就错了"
- **Episodic（情节型）**：会话摘要——"学生上次问了勾股定理逆定理的证明，理解后问了应用题"

### 如何启用

1. 在 [evermind.ai](https://evermind.ai) 免费注册，获取 API Key
2. 在 `.env` 中填写三行配置：
   ```bash
   EVERMEMOS_API_KEY=your-api-key-here
   EVERMEMOS_BASE_URL=https://api.evermind.ai
   EVERMEMOS_USER_ID=xiaomeng_studybuddy   # 改成学生名字，保持唯一
   ```
3. 重启服务器，后台自动工作

> 不配置则静默跳过，其余功能完全正常。EverMemOS 是可选的渐进增强，不是强依赖。

---

## 技术架构

### 数据流

```
手机拍照
   │
   ▼
[前端 index.html]        Alpine.js 单文件 SPA，无需 npm / 编译
   │  POST /api/v1/homework/grade
   ▼
[FastAPI 后端]
   │
   ▼ 串行 Agent 流水线
   ├─ OcrAgent         图片 → 识别题目文字
   ├─ GradeAgent       逐题判断对错 + 给出解析
   ├─ KnowPointAgent   标注知识点 + 难度等级
   └─ ExamTagAgent     分析试卷类型/来源
   │
   ├─→ WrongBookService     错题写入本地 JSON
   └─→ EverMemOSService     学情异步上传至云端记忆（fire-and-forget）
   │
   ▼
返回 JSON → 前端渲染批改结果
```

### 项目结构

```
StudyBuddy/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app、中间件、路由注册
│   │   ├── run_server.py        # 启动入口（直接运行这个）
│   │   └── routers/             # HTTP 路由
│   │       ├── homework.py      # POST /api/v1/homework/grade
│   │       ├── explain.py       # POST /api/v1/explain
│   │       ├── wrong_book.py    # GET/POST /api/v1/wrong-book
│   │       ├── memory.py        # GET /api/v1/profile
│   │       └── settings.py      # GET/POST /api/v1/settings
│   │
│   ├── agents/
│   │   ├── base_agent.py        # 所有 Agent 的父类（统一 LLM 调用、日志）
│   │   ├── homework/            # 批改流水线（OCR → Grade → KnowPoint → ExamTag）
│   │   ├── explain/             # 多轮问答 Agent（支持历史记忆注入）
│   │   └── memory/              # 用户画像更新 Agent
│   │
│   └── services/
│       ├── llm/                 # LLM 统一接口（自动路由到 OpenAI / Anthropic 等）
│       ├── evermemos/           # EverMemOS 客户端 + 业务逻辑层
│       ├── rag/                 # RAG 向量检索（教材问答用）
│       └── embedding/           # 向量化服务
│
├── web/
│   └── index.html               # 整个前端（Alpine.js 单文件 SPA）
│
├── config/
│   ├── main.yaml                # 全局配置
│   └── agents.yaml              # 各 Agent 的 LLM 参数
│
├── data/                        # 运行时数据（已 gitignore，自动生成）
├── requirements.txt
├── .env.example                 # 配置模板（安全，可提交）
└── .env                         # 你的真实配置（已 gitignore，勿提交）
```

### LLM 兼容性

`.env` 里配什么 Provider，代码自动适配，无需改代码：

| `LLM_BINDING` | 适用场景 |
|--------------|---------|
| `openai` | OpenAI、Gemini（v1beta兼容接口）、Kimi、DeepSeek 等 |
| `anthropic` | Anthropic Claude 系列 |

---

## API 文档

启动后访问 Swagger UI：`http://localhost:8001/docs`

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/homework/grade` | 上传作业图片，返回批改结果 |
| `POST` | `/api/v1/explain` | 多轮 AI 问答（支持图片） |
| `GET` | `/api/v1/wrong-book/list` | 获取错题列表 |
| `POST` | `/api/v1/wrong-book/mastered` | 标记错题已掌握 |
| `GET` | `/api/v1/profile` | 获取用户学情画像 |
| `GET/POST` | `/api/v1/settings` | 读写偏好设置 |

---

## 手机安装（PWA）

在 iOS Safari 中打开 `http://你的IP:8001/ui/`，点击底部分享按钮 → **添加到主屏幕**，即可全屏安装为本地 App。

---
