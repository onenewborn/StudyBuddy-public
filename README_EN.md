# StudyBuddy — AI Homework Assistant

A full-stack AI learning platform for middle school students. Snap a photo of your homework → AI grades it → wrong answers auto-archived → knowledge gaps tracked → personalized review.

Powered by **EverMemOS** long-term memory cloud, so the AI tutor truly *remembers* each student across sessions — grading history, weak knowledge points, learning preferences — getting smarter about you over time.

---

## Demo

<video src="https://github.com/user-attachments/assets/eb99171f-246f-4a90-99e0-91e803813c4b" controls width="100%"></video>

---

## Features

- **AI Homework Grading**: Upload a photo, AI identifies each question, marks correct/wrong, gives the correct answer and explains the mistake
- **Knowledge Point Tagging**: Every question is automatically tagged with relevant knowledge points and difficulty level; questions that hit the student's known weak areas are highlighted
- **Wrong Answer Notebook**: Incorrect answers are automatically archived, filterable by subject/topic, with one-click AI explanation and flashcard review
- **AI Q&A**: Multi-turn conversation for difficult questions; the AI draws on the student's learning history to give personalized explanations
- **Learning Analytics**: Automatically tracks accuracy rate, error type distribution, and weak knowledge point trends by subject

---

## Deployment

### Requirements

- Python 3.10+
- Any LLM API with vision input support (recommended: Gemini 2.5 Pro or GPT-4o)
- (Optional) EverMemOS API Key for cross-session long-term memory

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/StudyBuddy.git
cd StudyBuddy
```

### Step 2: Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Copy the template and fill in your API keys:

```bash
cp .env.example .env
```

Open `.env` and fill in at least the required fields:

```bash
# Primary LLM — must support image/vision input (used for OCR and grading)
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
LLM_HOST=https://api.openai.com/v1

# EverMemOS long-term memory (optional — skip to disable memory features)
EVERMEMOS_API_KEY=your-api-key
EVERMEMOS_USER_ID=my_studybuddy
```

See [`.env.example`](.env.example) for the full list of configuration options.

> **Recommended model**: Gemini 2.5 Pro — great vision accuracy, cost-effective, and supports the OpenAI-compatible API.
> Set `LLM_BINDING=openai` and `LLM_HOST=https://generativelanguage.googleapis.com/v1beta/openai/`.

### Step 4: Start the Server

```bash
python src/api/run_server.py
```

The server listens on `http://0.0.0.0:8001` by default. Change the port via `BACKEND_PORT` in `.env`.

### Step 5: Open the App

In your browser:

```
http://localhost:8001/ui/
```

**Mobile access (LAN)**: Make sure your phone and computer are on the same Wi-Fi, then use your computer's local IP:

```
http://192.168.x.x:8001/ui/
```

Find your local IP:

```bash
# macOS / Linux
ip addr show | grep "inet " | grep -v 127.0.0.1

# macOS shortcut
ipconfig getifaddr en0
```

---

## EverMemOS: Long-Term Memory That Actually Knows Your Student

### Why does long-term memory matter?

Standard AI has no memory — every conversation starts from scratch. Even after a month of using StudyBuddy, the AI won't know that this particular student has always struggled with factoring polynomials, or where they got stuck the last time they asked for help.

[EverMemOS](https://evermind.ai) solves this. It's a cloud-based long-term memory service built for AI applications. StudyBuddy automatically writes learning data to EverMemOS after each grading session. Before the AI responds next time, it retrieves relevant memories and delivers genuinely personalized explanations.

### What gets written to memory?

| When | What is stored |
|------|---------------|
| Every homework grading session | Subject, question count, accuracy rate, error type breakdown |
| Every wrong answer | Question text, student's answer, correct answer, error type |
| Knowledge gap detected | List of weak knowledge points, stored as a "review reminder" |
| AI conversation session compressed | Session summary (what the student asked + AI's learning analysis) |
| Student updates profile settings | Nickname, grade, MBTI, explanation style preference |

### Where does memory make a difference?

**During grading (KnowPointAgent)**: The student's historical weak knowledge points are injected as context. Questions that match known gaps are flagged with `is_weak_area=true`, so students and teachers can prioritize what actually needs reinforcement.

**During Q&A (ExplainAgent)**: Before responding, the AI retrieves the student's learning memories — explanation style preference, recent weak points, common error patterns — and tailors its answer accordingly, instead of giving a one-size-fits-all response.

### Memory types

EverMemOS organizes memories into three categories, all used by StudyBuddy:

- **EventLog (factual)**: Grading records and wrong answer details — *"Last week the student scored 72% on math, missing 3 factoring questions"*
- **Foresight (predictive)**: Review reminders for weak points — *"Next time this student encounters the difference of squares formula, flag it — they got it wrong last time"*
- **Episodic (narrative)**: Session summaries — *"Student asked about the converse of the Pythagorean theorem, understood after explanation, then asked a word problem"*

### How to enable

1. Sign up for free at [evermind.ai](https://evermind.ai) and get your API Key
2. Add three lines to `.env`:
   ```bash
   EVERMEMOS_API_KEY=your-api-key-here
   EVERMEMOS_BASE_URL=https://api.evermind.ai
   EVERMEMOS_USER_ID=xiaomeng_studybuddy   # use any unique identifier per student
   ```
3. Restart the server — everything works automatically in the background

> If not configured, all memory writes are silently skipped and everything else works normally. EverMemOS is a progressive enhancement, not a hard dependency.

---

## Architecture

### Data Flow

```
Student takes photo
        │
        ▼
[Frontend index.html]      Alpine.js single-file SPA — no npm, no build step
        │  POST /api/v1/homework/grade
        ▼
[FastAPI Backend]
        │
        ▼  Sequential Agent pipeline
        ├─ OcrAgent          Image → extract question text
        ├─ GradeAgent        Grade each question + explain errors
        ├─ KnowPointAgent    Tag knowledge points + difficulty level
        └─ ExamTagAgent      Analyze exam type and source
        │
        ├─→ WrongBookService       Save wrong answers to local JSON
        └─→ EverMemOSService       Upload learning data to cloud memory (fire-and-forget)
        │
        ▼
Return JSON → frontend renders grading results
```

### Project Structure

```
StudyBuddy/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app, middleware, router registration
│   │   ├── run_server.py        # Server entry point (run this)
│   │   └── routers/             # HTTP route handlers
│   │       ├── homework.py      # POST /api/v1/homework/grade
│   │       ├── explain.py       # POST /api/v1/explain
│   │       ├── wrong_book.py    # GET/POST /api/v1/wrong-book
│   │       ├── memory.py        # GET /api/v1/profile
│   │       └── settings.py      # GET/POST /api/v1/settings
│   │
│   ├── agents/
│   │   ├── base_agent.py        # Parent class for all agents (unified LLM calls, logging)
│   │   ├── homework/            # Grading pipeline (OCR → Grade → KnowPoint → ExamTag)
│   │   ├── explain/             # Multi-turn Q&A agent (with memory injection)
│   │   └── memory/              # User profile update agent
│   │
│   └── services/
│       ├── llm/                 # Unified LLM interface (auto-routes to OpenAI / Anthropic / etc.)
│       ├── evermemos/           # EverMemOS HTTP client + business logic layer
│       ├── rag/                 # RAG vector retrieval (for textbook Q&A)
│       └── embedding/           # Embedding service
│
├── web/
│   └── index.html               # Entire frontend (Alpine.js single-file SPA)
│
├── config/
│   ├── main.yaml                # Global configuration
│   └── agents.yaml              # Per-agent LLM parameters (temperature, max_tokens)
│
├── data/                        # Runtime data (gitignored, auto-generated)
├── requirements.txt
├── .env.example                 # Config template (safe to commit)
└── .env                         # Your real config with API keys (gitignored — never commit)
```

### LLM Compatibility

Set your provider in `.env` — the code adapts automatically, no code changes needed:

| `LLM_BINDING` | Works with |
|--------------|-----------|
| `openai` | OpenAI, Gemini (v1beta OpenAI-compatible endpoint), Kimi, DeepSeek, and more |
| `anthropic` | Anthropic Claude models |

---

## API Reference

Swagger UI available at `http://localhost:8001/docs` after startup.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/homework/grade` | Upload homework image, returns grading result |
| `POST` | `/api/v1/explain` | Multi-turn AI Q&A (supports image input) |
| `GET` | `/api/v1/wrong-book/list` | Fetch wrong answer list |
| `POST` | `/api/v1/wrong-book/mastered` | Mark a wrong answer as mastered |
| `GET` | `/api/v1/profile` | Get student learning profile |
| `GET/POST` | `/api/v1/settings` | Read / update preference settings |

---

## Mobile Installation (PWA)

Open `http://your-IP:8001/ui/` in iOS Safari, tap the Share button → **Add to Home Screen**. The app launches full-screen just like a native app.

---

## FAQ

**Q: Which LLM gives the best results?**
A: Gemini 2.5 Pro — excellent vision accuracy for handwritten homework, cost-effective, and uses the OpenAI-compatible interface. GPT-4o is also great but pricier. See `.env.example` for setup.

**Q: Can I use it without EverMemOS?**
A: Yes. Leave `EVERMEMOS_API_KEY` empty and all memory operations are silently skipped. Every other feature works normally.

**Q: What subjects are supported?**
A: Math, Physics, Chemistry, English, Chinese, History, Biology, Politics, Geography. The default system prompts are tuned for Grade 8 (middle school). You can adjust the grade level in the system prompt inside `src/agents/homework/grade_agent.py`.

**Q: Where is the student data stored?**
A: Everything is stored locally in JSON files under `data/`. No database required. EverMemOS is an optional cloud supplement — not a replacement for local storage.
