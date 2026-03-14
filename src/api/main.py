"""
StudyBuddy FastAPI Application
================================

启动方式：
    python src/api/run_server.py

路由前缀 /api/v1
    /homework   — 作业批改（OCR → 批改 → 知识点 → 试卷标签）
    /wrong-book — 错题本（增删改查 + 统计）
    /explain    — 难题解析（理科纯 LLM；文科 RAG + LLM，教材 ground truth）
    /settings   — 界面设置
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import explain, history, homework, memory, settings, textbook, wrong_book
from src.logging import get_logger

logger = get_logger("API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理（启动 + 关闭）。"""
    logger.info("StudyBuddy API starting up")

    # 预热 LLM 客户端，让 .env 中的配置提前加载到环境变量
    try:
        from src.services.llm import get_llm_client

        llm_client = get_llm_client()
        logger.info(f"LLM client ready: model={llm_client.config.model}")
    except Exception as e:
        logger.warning(f"LLM client init failed (will retry on first request): {e}")

    yield

    logger.info("StudyBuddy API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="StudyBuddy API",
    description="初中二年级学习助手 — 作业批改 / 知识库 / 设置",
    version="1.0.0",
    lifespan=lifespan,
    # 避免 HTTPS 反代时 307 重定向降级为 HTTP
    redirect_slashes=False,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 生产环境改为具体前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静态文件（仅暴露作业图片目录，避免 data/ 下敏感 JSON 文件被公开访问）──────
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

hw_images_dir = data_dir / "homework_images"
hw_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/outputs/homework_images", StaticFiles(directory=str(hw_images_dir)), name="outputs")

# ── 前端静态文件 ──────────────────────────────────────────────────────────────
web_dir = project_root / "web"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(homework.router,    prefix="/api/v1/homework",    tags=["homework"])
app.include_router(wrong_book.router,  prefix="/api/v1/wrong-book",  tags=["wrong-book"])
app.include_router(explain.router,     prefix="/api/v1/explain",     tags=["explain"])
app.include_router(settings.router,    prefix="/api/v1/settings",    tags=["settings"])
app.include_router(memory.router,      prefix="/api/v1/profile",     tags=["profile"])
app.include_router(history.router,     prefix="/api/v1/history",     tags=["history"])
app.include_router(textbook.router,                                   tags=["textbook"])
# knowledge router 暂不注册（依赖 src.api.utils，StudyBuddy 尚未实现）
# app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Welcome to StudyBuddy API", "version": "1.0.0"}


@app.get("/api/v1/health")
async def health():
    """全局健康检查。"""
    return {"status": "ok"}
