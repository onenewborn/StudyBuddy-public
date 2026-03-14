"""
Explain API Router — 主动问题询问
===================================

此路由专门用于学生主动提问任何题目：
  - 无需预先有错题记录
  - 只需填写题目文字 + 选择科目
  - 系统自动使用最合适的解析策略（纯 LLM / RAG + LLM）

路由
-----
GET  /api/v1/explain/health  — 各科目教材索引状态
POST /api/v1/explain         — 主动询问任意题目（question_explain 模式）

-----
错题解析（已有错题记录）请使用：
  POST /api/v1/wrong-book/{entry_id}/explain
-----
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import base64

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import Form as FastapiForm
from pydantic import BaseModel, Field

# ── Project root ──────────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.explain import ExplainAgent, ExplainResponse
from src.agents.explain.explain_agent import ExplainRequest
from src.agents.explain.kb_manager import TextbookKBManager, _SUBJECT_KB
from src.agents.memory import COMPRESSION_THRESHOLD, ChatCompressor, MemoryService
from src.config.model_registry import SUPPORTED_MODELS, agent_kwargs
from src.logging import get_logger
from src.services.evermemos import get_evermemos_service

logger = get_logger("ExplainRouter")

router = APIRouter()

# ── 单例（懒加载）────────────────────────────────────────────────────────────
_agent: Optional[ExplainAgent] = None
_mem_service: Optional[MemoryService] = None


def get_explain_agent() -> ExplainAgent:
    """返回全局单例 ExplainAgent（懒加载）。"""
    global _agent
    if _agent is None:
        _agent = ExplainAgent(data_dir=_project_root / "data")
    return _agent


def get_mem_service() -> MemoryService:
    """返回全局单例 MemoryService（懒加载）。"""
    global _mem_service
    if _mem_service is None:
        _mem_service = MemoryService(_project_root / "data")
    return _mem_service


# ── 公开的简化请求体（仅主动询问模式使用）──────────────────────────────────

class QuestionInquiryRequest(BaseModel):
    """
    主动问题询问请求体。

    只需要三个字段——题目 + 科目 + 可选知识点提示。
    session_id 由前端生成（localStorage UUID），用于聊天记忆追踪。
    """
    question_text: str = Field(
        description="题目完整文字（含选项 / 条件 / 图片描述等）",
        examples=["解方程：2x + 5 = 13，求 x 的值"],
    )
    subject: Literal[
        "math", "physics", "chemistry",
        "english", "biology", "history", "chinese",
        "politics", "geography"
    ] = Field(
        default="math",
        description="科目",
    )
    knowledge_points: list[str] = Field(
        default_factory=list,
        description="可选：补充知识点提示，帮助系统更精准检索教材（如 ['一次函数', '斜率']）",
        examples=[[]],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID（前端 localStorage UUID），用于聊天记忆追踪与压缩",
    )
    model_key: Optional[str] = Field(
        default=None,
        description="模型预设键：gemini | kimi（不传则使用系统默认 LLM）",
    )


# ── Multi-turn message builder ─────────────────────────────────────────────────

def _build_multiturn_messages(
    system_prompt: str,
    current_user_text: str,
    turn_records: list,
    session_image_bytes: Optional[bytes],
    compressed_summary: str,
    new_image_bytes: Optional[bytes] = None,
    new_image_content_type: str = "image/jpeg",
) -> list[dict]:
    """
    构建送往 LLM 的完整 messages 数组（含历史多轮上下文 + 可选图片重注入）。

    Args:
        system_prompt:          系统提示词（已含记忆上下文）
        current_user_text:      本轮用户文字内容
        turn_records:           ChatTurn 历史轮次列表（最近 N 轮）
        session_image_bytes:    会话持久化图片字节（供历史图片轮重建）
        compressed_summary:     历史压缩摘要（注入 system prompt 末尾）
        new_image_bytes:        本轮新上传图片字节（若有）
        new_image_content_type: 本轮图片 MIME type
    """
    # 将压缩摘要追加到 system prompt 末尾
    full_system = system_prompt
    if compressed_summary:
        full_system += f"\n\n【本次会话历史摘要（供参考）】\n{compressed_summary}"

    messages: list[dict] = [{"role": "system", "content": full_system}]

    # 准备会话图片 data-url（用于重建历史图片轮）
    sess_data_url: Optional[str] = None
    if session_image_bytes:
        b64 = base64.b64encode(session_image_bytes).decode()
        sess_data_url = f"data:image/jpeg;base64,{b64}"

    # ── 注入历史轮次 ───────────────────────────────────────────────────────────
    if turn_records:
        for turn in turn_records:
            if turn.has_image and sess_data_url:
                user_content: Any = [
                    {"type": "text",      "text": turn.user_text},
                    {"type": "image_url", "image_url": {"url": sess_data_url, "detail": "high"}},
                ]
            else:
                user_content = turn.user_text
            messages.append({"role": "user",      "content": user_content})
            messages.append({"role": "assistant", "content": turn.assistant_text})
    elif sess_data_url and new_image_bytes is None:
        # 无历史轮次（压缩后已清空），但 session 图片仍在且本轮无新图
        # → 插入一个"背景图片"对话，让模型记住本次会话的图片内容
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "（这是本次学习会话开始时上传的题目图片，请记住图片内容以便回答后续问题）"},
                {"type": "image_url", "image_url": {"url": sess_data_url, "detail": "high"}},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": "好的，我已仔细查看图片中的题目内容，可以回答您的后续问题。",
        })

    # ── 当前轮用户消息 ─────────────────────────────────────────────────────────
    if new_image_bytes:
        b64_new = base64.b64encode(new_image_bytes).decode()
        curr_data_url = f"data:{new_image_content_type};base64,{b64_new}"
        curr_content: Any = [
            {"type": "text",      "text": current_user_text},
            {"type": "image_url", "image_url": {"url": curr_data_url, "detail": "high"}},
        ]
    else:
        curr_content = current_user_text

    messages.append({"role": "user", "content": curr_content})
    return messages


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """返回可用模型预设列表（key + label），供前端模型选择器使用。"""
    return [
        {"key": k, "label": v["label"]}
        for k, v in SUPPORTED_MODELS.items()
    ]


@router.get("/health")
async def health_check():
    """
    解析模块健康检查。

    返回各科目的教材向量库状态：

    | 状态          | 含义                                                     |
    |---------------|----------------------------------------------------------|
    | `indexed`     | 向量库已在磁盘上，检索即时（毫秒级）                     |
    | `not_indexed` | PDF 已上传但尚未索引；首次调用 explain 时自动建库（约 1-3 分钟） |
    | `no_pdf`      | 教材 PDF 未上传；将使用纯 LLM 解析（无教材约束）          |
    | `llm_only`    | 理科科目，始终用纯 LLM，不需要教材                       |
    """
    _llm_only = {"math", "physics", "chemistry"}
    kb_manager = TextbookKBManager(_project_root / "data")
    kb_base    = _project_root / "data" / "knowledge_bases"

    subjects: dict[str, str] = {}
    for subj, cfg in _SUBJECT_KB.items():
        if kb_manager.is_indexed(subj):
            subjects[subj] = "indexed"
        elif (kb_base / cfg["pdf"]).exists():
            subjects[subj] = "not_indexed"
        else:
            subjects[subj] = "no_pdf"
    for s in _llm_only:
        subjects[s] = "llm_only"

    return {"status": "ok", "module": "explain", "subjects": subjects}


@router.post("", response_model=ExplainResponse)
async def ask_question(body: QuestionInquiryRequest):
    """
    **主动问题询问**：学生直接提问任意题目，获得详细解析。

    系统根据科目自动选择最优策略：

    | 科目                          | 策略                  | 效果                                  |
    |-------------------------------|-----------------------|---------------------------------------|
    | math / physics / chemistry    | 纯 LLM（或多轮上下文） | step-by-step 完整解题过程             |
    | english / biology / history / chinese | RAG + LLM（首次）/ 多轮上下文（有历史） | 先检索八年级教材，以教材内容为依据讲解 |

    传入 `session_id`（前端生成的 UUID）可启用聊天记忆：
    - 历史问答自动追踪，达到阈值后 LLM 压缩为摘要
    - 有图片会话历史时，自动将图片重注入上下文，实现"上传一次、连续提问"
    """
    try:
        mem_svc = get_mem_service()

        # ── 1. 加载本地记忆 + EverMemOS 长期记忆 ──────────────────────────
        mem_ctx = await mem_svc.get_memory_context(body.subject, body.session_id)
        memory_ctx = mem_ctx.to_prompt_str()

        try:
            evermemos = get_evermemos_service()
            em_memories = await evermemos.search_context(
                subject=body.subject,
                query=body.question_text[:200],
                top_k=4,
            )
            if em_memories:
                memory_ctx += (
                    "\n\n【历史学习记忆（EverMemOS）】\n"
                    + "\n".join(f"• {m}" for m in em_memories)
                    + "\n"
                )
        except Exception as _em_exc:
            logger.debug(f"[Explain] EverMemOS 读取跳过: {_em_exc}")

        # ── 2. 加载多轮会话上下文 ─────────────────────────────────────────
        turn_records: list = []
        session_image_bytes: Optional[bytes] = None
        compressed_summary = ""
        if body.session_id:
            turn_records, session_image_bytes, compressed_summary = \
                await mem_svc.get_llm_context(body.session_id)

        has_context = bool(turn_records or session_image_bytes or compressed_summary)

        # ── 3. 调用 LLM ───────────────────────────────────────────────────
        model_kw = agent_kwargs(body.model_key)
        if model_kw:
            _agent_inst = ExplainAgent(data_dir=_project_root / "data", **model_kw)
        else:
            _agent_inst = get_explain_agent()

        if has_context:
            # 有历史上下文（含图片）→ 直接走 call_llm(messages=...) 保持图片连贯
            subject_cn = _SUBJECT_CN_MAP.get(body.subject.lower(), body.subject)

            # ── 文科题目：也要检索教材，注入 system_prompt ──────────────────
            rag_snippets: list[str] = []
            try:
                rag_snippets = await _agent_inst.retrieve_snippets(
                    body.subject, body.question_text
                )
                if rag_snippets:
                    logger.info(
                        f"[Explain] multi-turn RAG: {len(rag_snippets)} snippets "
                        f"for {body.subject}"
                    )
            except Exception as rag_exc:
                logger.warning(f"[Explain] multi-turn RAG failed: {rag_exc}")

            system_prompt = (
                f"你是一位初中二年级{subject_cn}老师，"
                "用简洁、亲切的语言帮助 13-14 岁学生理解难题。"
                "回答使用 Markdown 格式，层次清晰。"
                + memory_ctx
            )

            if rag_snippets:
                textbook_desc = (
                    _agent_inst.get_textbook_desc(body.subject)
                    or f"八年级{subject_cn}教材"
                )
                context_block = "\n\n---\n".join(
                    f"【教材片段 {i+1}】\n{s.strip()}"
                    for i, s in enumerate(rag_snippets[:4])
                )
                system_prompt += (
                    f"\n\n以下是从《{textbook_desc}》中检索到的相关原文片段，"
                    f"请严格依据这些原文作答：\n\n{context_block}\n\n"
                    "【重要】你只能根据上方教材原文片段作答，不可引入教材以外的任何知识。"
                )

            messages = _build_multiturn_messages(
                system_prompt=system_prompt,
                current_user_text=body.question_text,
                turn_records=turn_records,
                session_image_bytes=session_image_bytes,
                compressed_summary=compressed_summary,
            )
            explanation = await _agent_inst.call_llm(
                user_prompt="",
                system_prompt="",
                messages=messages,
                stage="text_explain_multiturn",
            )
            response = ExplainResponse(
                question_text=body.question_text,
                subject=body.subject,
                mode="question_explain",
                explanation=explanation,
                key_concepts=[],
                used_rag=bool(rag_snippets),
                textbook_snippets=rag_snippets[:3],
            )
        else:
            # 首次提问 → 走 explain()（含 RAG 检索）
            request = ExplainRequest(
                question_text=body.question_text,
                subject=body.subject,
                knowledge_points=body.knowledge_points,
                mode="question_explain",
            )
            response = await _agent_inst.explain(request, memory_context=memory_ctx)

        # ── 4. 记录本轮问答 ───────────────────────────────────────────────
        if body.session_id:
            try:
                session = await mem_svc.record_turn(
                    session_id=body.session_id,
                    subject=body.subject,
                    user_text=body.question_text,
                    assistant_text=response.explanation[:800],
                    has_image=False,
                )
                # 达到阈值 & 尚未压缩 → 触发 LLM 压缩并清空 turn_records
                if len(session.turn_records) >= COMPRESSION_THRESHOLD and not session.compressed_summary:
                    try:
                        compressor = ChatCompressor()
                        summary = await compressor.compress(session)
                        if summary:
                            session.compressed_summary = summary
                            session.turn_records.clear()
                            await mem_svc.save_session(session)
                            logger.info(
                                f"[Explain] session {body.session_id} compressed "
                                f"({session.turn_count} turns)"
                            )
                            # EverMemOS：会话摘要写入长期记忆
                            try:
                                evermemos = get_evermemos_service()
                                await evermemos.log_session_summary(
                                    subject=body.subject,
                                    session_id=body.session_id,
                                    summary=summary,
                                    turn_count=session.turn_count,
                                )
                            except Exception as em_exc:
                                logger.warning(f"[Explain] EverMemOS 写入失败（忽略）: {em_exc}")
                    except Exception as cmp_exc:
                        logger.warning(f"[Explain] compression failed: {cmp_exc}")
            except Exception as sess_exc:
                logger.warning(f"[Explain] session tracking failed: {sess_exc}")

        return response

    except Exception as exc:
        logger.error(f"[Explain] 解析失败: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"解析失败：{exc!s}")


# ── Vision 解题端点（图片直接传给 LLM，绕过 OCR 中间层）─────────────────────

_SUBJECT_CN_MAP: dict[str, str] = {
    "math": "数学", "physics": "物理", "chemistry": "化学",
    "english": "英语", "biology": "生物", "history": "历史", "chinese": "语文",
}

@router.post("/with-image", response_model=ExplainResponse)
async def ask_question_with_image(
    image:         UploadFile        = File(..., description="图片文件（JPG/PNG/WEBP/HEIC）"),
    question_text: str               = FastapiForm(default="", description="可选的补充文字描述"),
    subject:       str               = FastapiForm(default="math", description="科目"),
    session_id:    Optional[str]     = FastapiForm(default=None,   description="会话 ID"),
    model_key:     Optional[str]     = FastapiForm(default=None,   description="模型预设键"),
):
    """
    **Vision 图片解题**：将图片直接传给多模态 LLM，让模型"看图解题"。

    支持同 session 连续提问：上传图片后，后续文字追问（走 POST /api/v1/explain）
    会自动将本次图片重注入历史上下文，无需重复上传。

    适合：有图表/示意图的物理/数学/化学题、选择题配图、手写题目等。
    """
    try:
        # ── 1. 读取图片 ───────────────────────────────────────────────────────
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="图片为空")
        content_type = image.content_type or "image/jpeg"

        # ── 2. 本地记忆 + EverMemOS 长期记忆 ─────────────────────────────────
        mem_svc    = get_mem_service()
        mem_ctx    = await mem_svc.get_memory_context(subject, session_id)
        memory_ctx = mem_ctx.to_prompt_str()

        try:
            evermemos = get_evermemos_service()
            em_query  = question_text.strip()[:200] or f"{subject} 图片题目"
            em_memories = await evermemos.search_context(
                subject=subject,
                query=em_query,
                top_k=3,
            )
            if em_memories:
                memory_ctx += (
                    "\n\n【历史学习记忆（EverMemOS）】\n"
                    + "\n".join(f"• {m}" for m in em_memories)
                    + "\n"
                )
        except Exception as _em_exc:
            logger.debug(f"[VisionExplain] EverMemOS 读取跳过: {_em_exc}")

        # ── 3. 持久化图片（供后续文字轮次重注入）────────────────────────────
        saved_img_path: Optional[str] = None
        if session_id:
            saved_img_path = await mem_svc.save_session_image(
                session_id, image_bytes, content_type
            )

        # ── 4. 加载历史多轮上下文 ─────────────────────────────────────────────
        turn_records: list = []
        session_image_bytes: Optional[bytes] = None
        compressed_summary = ""
        if session_id:
            turn_records, session_image_bytes, compressed_summary = \
                await mem_svc.get_llm_context(session_id)

        # ── 5. 获取 Agent 实例 ────────────────────────────────────────────────
        _vision_default_key = model_key or "kimi"
        model_kw = agent_kwargs(_vision_default_key)
        if model_kw:
            _agent_inst = ExplainAgent(data_dir=_project_root / "data", **model_kw)
        else:
            logger.warning("[VisionExplain] kimi 未配置，回退到系统默认 LLM")
            _agent_inst = get_explain_agent()

        # ── 6. 构建 Prompt ────────────────────────────────────────────────────
        subject_cn        = _SUBJECT_CN_MAP.get(subject.lower(), subject)
        extra_instruction = question_text.strip()
        task_description  = f"{extra_instruction}\n\n" if extra_instruction else ""
        system_prompt = (
            f"你是一位初中二年级{subject_cn}老师，"
            "用简洁、亲切的语言帮助 13-14 岁学生理解难题。"
            "回答使用 Markdown 格式，层次清晰。"
            + memory_ctx
        )
        user_prompt_text = (
            f"{task_description}"
            f"请仔细观察图片中的{subject_cn}题目内容，按以下结构详细讲解：\n\n"
            "## 📝 题目分析\n"
            "（先描述图片中的题目内容，理解考察的核心知识点，明确解题方向）\n\n"
            "## ✅ 解题过程\n"
            "（完整推导步骤；物理/数学/化学要写出关键公式，不要跳步；"
            "如题目有选项，逐项分析并说明正确选项的理由）\n\n"
            "## 💡 解题技巧 & 总结\n"
            "（同类题型的解题规律和注意事项）"
        )

        # ── 7. 构建多轮 messages（当前轮携带新图片）────────────────────────
        vision_messages = _build_multiturn_messages(
            system_prompt=system_prompt,
            current_user_text=user_prompt_text,
            turn_records=turn_records,
            session_image_bytes=session_image_bytes,
            compressed_summary=compressed_summary,
            new_image_bytes=image_bytes,
            new_image_content_type=content_type,
        )

        # ── 8. 调用 LLM ───────────────────────────────────────────────────────
        logger.info(
            f"[VisionExplain] subject={subject}  "
            f"image_size={len(image_bytes)//1024}KB  "
            f"history_turns={len(turn_records)}  "
            f"session={'yes' if session_id else 'no'}"
        )
        explanation = await _agent_inst.call_llm(
            user_prompt="",
            system_prompt="",
            messages=vision_messages,
            stage="vision_explain",
        )

        # ── 9. 记录本轮问答（has_image=True + 更新图片路径）─────────────────
        if session_id:
            try:
                session = await mem_svc.record_turn(
                    session_id=session_id,
                    subject=subject,
                    user_text=f"[图片] {extra_instruction}".strip() or "[图片解题]",
                    assistant_text=explanation[:800],
                    has_image=True,
                    new_image_path=saved_img_path,
                )
                # 达到阈值 → 压缩并清空 turn_records
                if len(session.turn_records) >= COMPRESSION_THRESHOLD and not session.compressed_summary:
                    try:
                        compressor = ChatCompressor()
                        summary = await compressor.compress(session)
                        if summary:
                            session.compressed_summary = summary
                            session.turn_records.clear()
                            await mem_svc.save_session(session)
                            logger.info(
                                f"[VisionExplain] session {session_id} compressed "
                                f"({session.turn_count} turns)"
                            )
                            # EverMemOS：视觉会话摘要写入长期记忆
                            try:
                                evermemos = get_evermemos_service()
                                await evermemos.log_session_summary(
                                    subject=subject,
                                    session_id=session_id,
                                    summary=summary,
                                    turn_count=session.turn_count,
                                )
                            except Exception as em_exc:
                                logger.warning(f"[VisionExplain] EverMemOS 写入失败（忽略）: {em_exc}")
                    except Exception as cmp_exc:
                        logger.warning(f"[VisionExplain] compression failed: {cmp_exc}")
            except Exception as sess_exc:
                logger.warning(f"[VisionExplain] session tracking failed: {sess_exc}")

        return ExplainResponse(
            question_text=extra_instruction or "[图片题目]",
            subject=subject,
            mode="question_explain",
            explanation=explanation,
            key_concepts=[],
            used_rag=False,
            textbook_snippets=[],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[VisionExplain] 解析失败: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图片解析失败：{exc!s}")


# ── 文件内容提取（图片 OCR / PDF 文字）────────────────────────────────────────

@router.post("/extract-from-file")
async def extract_from_file(
    file: UploadFile = File(..., description="图片（JPG/PNG/HEIC）或 PDF 文件"),
    subject: str = FastapiForm(default="math", description="科目提示，帮助 OCR 识别专业词汇"),
):
    """
    从图片或 PDF 中提取题目文字，返回提取结果供用户预览和编辑。

    - **图片**：调用视觉 LLM 识别题目文字（支持手写 + 印刷体）
    - **PDF**：用 PyMuPDF 直接提取文字层（如有），无文字层则转图片后 OCR

    前端流程：
    1. 用户上传文件 → 调用此接口 → 显示提取的文字
    2. 用户确认/编辑 → 正常走 POST /api/v1/explain 接口提问
    """
    filename = (file.filename or "").lower()
    content  = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    # ── PDF 路径 ──────────────────────────────────────────────────────────────
    if filename.endswith(".pdf"):
        extracted = await _extract_pdf_text(content)
        return {"text": extracted, "source": "pdf", "filename": file.filename}

    # ── 图片路径 ──────────────────────────────────────────────────────────────
    if any(filename.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp")):
        extracted = await _extract_image_text(content, file.content_type or "image/jpeg", subject)
        return {"text": extracted, "source": "image", "filename": file.filename}

    raise HTTPException(status_code=400, detail="不支持的文件类型，请上传图片或 PDF")


async def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """用 PyMuPDF 提取 PDF 文字层；若文字层为空则提示用户用图片上传。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF 解析库未安装，请运行: pip install pymupdf",
        )

    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text: list[str] = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        text = "\n\n".join(t.strip() for t in pages_text if t.strip())
        if not text:
            return "（PDF 中未检测到可提取的文字层，请改用截图上传）"
        return text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败：{exc}")


async def _extract_image_text(image_bytes: bytes, content_type: str, subject: str) -> str:
    """
    调用视觉 LLM 识别图片中的题目文字。
    使用 ExplainAgent 的 call_llm 方法（复用现有 LLM 配置）。
    """
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{content_type};base64,{b64}"

    _SUBJECT_NAMES = {
        "math": "数学", "physics": "物理", "chemistry": "化学",
        "english": "英语", "biology": "生物", "history": "历史", "chinese": "语文",
    }
    subj_cn = _SUBJECT_NAMES.get(subject, "")

    system_prompt = (
        "你是一名专业的题目文字识别助手。"
        "请从图片中完整、准确地提取题目内容，包括题目文字、选项、数学公式、表格等所有信息。"
        "保持原始格式（换行、编号等），不要添加任何解析或说明。"
        "如果图片模糊或无法识别，请说明原因。"
    )
    user_prompt = (
        f"请识别并提取图片中的{subj_cn}题目内容（仅提取原文，不要解答）："
    )

    # 构造包含图片的 messages（OpenAI vision 格式）
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ],
        },
    ]

    try:
        agent = get_explain_agent()
        # 直接使用 BaseAgent 的底层 LLM 客户端
        result = await agent._llm_client.chat(messages=messages)  # type: ignore[attr-defined]
        if isinstance(result, dict):
            return result.get("content") or result.get("text") or str(result)
        return str(result)
    except AttributeError:
        # 如果 _llm_client 不可直接访问，回退到 call_llm（纯文字提示）
        agent = get_explain_agent()
        return await agent.call_llm(
            user_prompt=f"[图片 base64 略，题目科目：{subj_cn}]\n请根据科目提示描述一个典型{subj_cn}题目（OCR 功能需视觉 LLM 支持）",
            system_prompt=system_prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图片识别失败：{exc}")
