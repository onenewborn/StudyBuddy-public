"""
Wrong Book API Router — 错题本
================================

Endpoints
---------
GET    /api/v1/wrong-book                    — 列出所有错题（支持 subject / mastered 过滤）
GET    /api/v1/wrong-book/stats              — 错题本统计摘要
GET    /api/v1/wrong-book/{id}               — 获取单条错题
POST   /api/v1/wrong-book/{id}/explain       — 一键解析该条错题（无需 body）
PUT    /api/v1/wrong-book/{id}/reviewed      — 标记已复习
PUT    /api/v1/wrong-book/{id}/mastered      — 设置掌握状态
DELETE /api/v1/wrong-book/{id}              — 删除一条错题
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pathlib import Path as _Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.config.model_registry import agent_kwargs

# ── Project root in path ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.explain import ExplainAgent, ExplainResponse
from src.agents.explain.explain_agent import ExplainRequest
from src.agents.homework.models import WrongBookEntry
from src.agents.homework.wrong_book_service import WrongBookService
from src.agents.memory import MemoryService
from src.logging import get_logger

logger = get_logger("WrongBook")

# ── Shared service instance（懒加载，首次调用时初始化）────────────────────────
_data_dir = _project_root / "data"
_service: Optional[WrongBookService] = None
_explain_agent: Optional[ExplainAgent] = None
_mem_service: Optional[MemoryService] = None


def get_service() -> WrongBookService:
    global _service
    if _service is None:
        _service = WrongBookService(_data_dir)
    return _service


def _get_explain_agent() -> ExplainAgent:
    global _explain_agent
    if _explain_agent is None:
        _explain_agent = ExplainAgent(data_dir=_data_dir)
    return _explain_agent


def _get_mem_service() -> MemoryService:
    global _mem_service
    if _mem_service is None:
        _mem_service = MemoryService(_data_dir)
    return _mem_service


# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response helpers
# ─────────────────────────────────────────────────────────────────────────────

class MasteredUpdate(BaseModel):
    mastered: bool


class OverrideCorrectRequest(BaseModel):
    question_text: str
    subject: str
    student_answer: str = ""


class WrongBookStats(BaseModel):
    total: int
    mastered: int
    unmastered: int
    by_subject: dict[str, int]
    by_difficulty: dict[str, int]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[WrongBookEntry])
async def list_entries(
    subject: Optional[str] = Query(
        default=None,
        description="按科目过滤，如 math / physics / chinese",
    ),
    mastered: Optional[bool] = Query(
        default=None,
        description="true = 只看已掌握；false = 只看未掌握；不传 = 全部",
    ),
):
    """
    列出错题本条目。

    - 不传任何参数 → 返回全部（按创建时间倒序）
    - `subject=math` → 只返回数学错题
    - `mastered=false` → 只返回尚未掌握的题目
    """
    svc = get_service()
    return await svc.list_entries(subject=subject, mastered=mastered)


@router.get("/stats", response_model=WrongBookStats)
async def get_stats():
    """
    错题本统计摘要：总数、已掌握数、按科目分布、按难度分布。
    """
    svc = get_service()
    stats = await svc.get_stats()
    return WrongBookStats(**stats)


@router.post("/{entry_id}/explain", response_model=ExplainResponse)
async def explain_entry(
    entry_id: str,
    model_key: Optional[str] = Query(
        default=None,
        description="模型预设键：gemini | kimi（不传则使用系统默认）",
    ),
):
    """
    **错题一键解析**：无需填写任何 body，系统自动加载错题上下文并生成详细解析。

    后端会自动读取该条错题的所有信息（题目、学生答案、正确答案、知识点、错误类型等），
    并以「错题解析模式」调用 AI 解析，重点分析：

    1. 正确答案的完整解题过程
    2. 学生答案为什么错（针对具体错误类型分析）
    3. 同类题型的解题技巧

    文科/语言类科目会额外检索八年级教材，确保解析在课标范围内。

    **调用示例**：
    ```
    POST /api/v1/wrong-book/english_20250228_143021_1/explain
    ```
    （无 request body）
    """
    # 1. 加载错题
    svc = get_service()
    entry = await svc.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")

    # 2. 将 WrongBookEntry 映射到 ExplainRequest（error_analysis 模式）
    request = ExplainRequest(
        question_text=entry.question_text,
        subject=entry.subject,
        student_answer=entry.student_answer,
        correct_answer=entry.correct_answer,
        question_type=entry.question_type.value,
        knowledge_points=list(entry.knowledge_points),
        error_type=entry.error_type.value if entry.error_type else None,
        brief_comment=entry.brief_comment,
        mode="error_analysis",
    )

    # 3. 尝试加载原始作业图片
    image_bytes: Optional[bytes] = None
    image_content_type = "image/jpeg"
    if entry.source_image_path:
        img_path = _Path(entry.source_image_path)
        if img_path.exists():
            try:
                image_bytes = img_path.read_bytes()
                # 根据文件后缀推断 MIME type
                sfx = img_path.suffix.lower()
                if sfx in (".png",):
                    image_content_type = "image/png"
                elif sfx in (".webp",):
                    image_content_type = "image/webp"
            except Exception as img_exc:
                logger.warning(f"[WrongBook] 读取图片失败（降级为纯文字解析）: {img_exc}")

    # 4. 解析
    try:
        model_kw = agent_kwargs(model_key)
        explain_agent = (
            ExplainAgent(data_dir=_data_dir, **model_kw)
            if model_kw
            else _get_explain_agent()
        )
        if image_bytes:
            logger.info(
                f"[WrongBook] 视觉解析 entry={entry_id} "
                f"image={len(image_bytes)//1024}KB q={entry.question_number!r}"
            )
            return await explain_agent.explain_with_image(
                request,
                image_bytes=image_bytes,
                question_number=entry.question_number or "",
                content_type=image_content_type,
            )
        return await explain_agent.explain(request)
    except Exception as exc:
        logger.error(f"[WrongBook] 解析失败 entry={entry_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"解析失败：{exc!s}")


@router.get("/{entry_id}", response_model=WrongBookEntry)
async def get_entry(entry_id: str):
    """获取指定 ID 的单条错题。"""
    svc = get_service()
    entry = await svc.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    return entry


@router.put("/{entry_id}/reviewed", response_model=WrongBookEntry)
async def mark_reviewed(entry_id: str):
    """
    标记此题已复习一次：review_count +1，last_reviewed_at 更新为当前时间。
    """
    svc = get_service()
    entry = await svc.mark_reviewed(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    return entry


@router.put("/{entry_id}/mastered", response_model=WrongBookEntry)
async def set_mastered(entry_id: str, body: MasteredUpdate):
    """
    设置掌握状态。

    - `{"mastered": true}` → 标记为已掌握
    - `{"mastered": false}` → 取消掌握（重新归入待复习）
    """
    svc = get_service()
    entry = await svc.mark_mastered(entry_id, body.mastered)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    return entry


@router.post("/override-correct")
async def override_correct(body: OverrideCorrectRequest):
    """
    **学生申诉覆盖**：将某道被 AI 判为错误的题目标记为「实际正确」。

    操作流程：
    1. 按题目文字 + 科目 + 学生答案在错题本中查找对应条目
    2. 找到则删除该条目（题目从错题本移除）
    3. 调整最近一条该科目成绩记录（wrong_count -1, correct_count +1）

    - 若未在错题本找到匹配条目（可能已删除），仍尝试调整成绩记录
    - 返回 `{deleted_entry_id, adjusted_record}` 说明操作结果
    """
    svc = get_service()
    mem_svc = _get_mem_service()

    # 1. 查找错题条目
    entry_id = await svc.find_entry_by_question(
        question_text=body.question_text,
        subject=body.subject,
        student_answer=body.student_answer,
    )

    # 2. 删除错题本条目（如果找到）
    deleted_id: Optional[str] = None
    if entry_id:
        deleted = await svc.delete_entry(entry_id)
        if deleted:
            deleted_id = entry_id
            logger.info(f"[WrongBook] 申诉覆盖：已删除错题 {entry_id}")

    # 3. 调整成绩记录
    adjusted = await mem_svc.adjust_last_record(subject=body.subject)

    return {
        "ok": True,
        "deleted_entry_id": deleted_id,
        "adjusted_record": adjusted,
    }


@router.delete("/{entry_id}")
async def delete_entry(entry_id: str):
    """
    永久删除一条错题。此操作不可恢复。
    """
    svc = get_service()
    deleted = await svc.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"错题 '{entry_id}' 不存在")
    return {"deleted": True, "entry_id": entry_id}
