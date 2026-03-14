"""
Profile / Memory API Router
============================

路由前缀：/api/v1/profile

Endpoints
---------
GET    /api/v1/profile                        — 读取完整用户记忆
PATCH  /api/v1/profile/preferences            — 更新偏好设置
GET    /api/v1/profile/performance            — 成绩记录列表
GET    /api/v1/profile/performance/summary    — 成绩趋势 + 考试焦虑分析
GET    /api/v1/profile/streak                 — 连续学习天数
GET    /api/v1/profile/knowledge-gaps         — 各科薄弱知识点
GET    /api/v1/profile/review-queue           — 优先复习科目建议
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.memory import MemoryService, UserMemory
from src.logging import get_logger

logger = get_logger("ProfileRouter")

router = APIRouter()

_data_dir = _project_root / "data"
_mem_service: Optional[MemoryService] = None


def get_mem_service() -> MemoryService:
    global _mem_service
    if _mem_service is None:
        _mem_service = MemoryService(_data_dir)
    return _mem_service


# ── Request / Response models ──────────────────────────────────────────────────

class PreferencesUpdate(BaseModel):
    explanation_length: Optional[str] = None   # brief | standard | detailed
    explanation_style:  Optional[str] = None   # vivid | objective
    nickname:           Optional[str] = None
    grade:              Optional[str] = None
    gender:             Optional[str] = None   # male | female | other
    mbti:               Optional[str] = None   # MBTI 16-type, e.g. INTP


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("", response_model=UserMemory)
async def get_profile():
    """读取用户完整记忆档案（偏好、成绩、薄弱知识点、连续学习天数…）。"""
    return await get_mem_service().load_memory()


@router.patch("/preferences")
async def update_preferences(body: PreferencesUpdate):
    """
    更新解释偏好。

    - `explanation_length`：brief（简短）| standard（适中）| detailed（详细）
    - `explanation_style`：vivid（生动形象）| objective（简洁客观）
    """
    kwargs: dict = {}
    if body.explanation_length is not None:
        if body.explanation_length not in {"brief", "standard", "detailed"}:
            raise HTTPException(400, "explanation_length 必须是 brief / standard / detailed")
        kwargs["explanation_length"] = body.explanation_length
    if body.explanation_style is not None:
        if body.explanation_style not in {"vivid", "objective"}:
            raise HTTPException(400, "explanation_style 必须是 vivid / objective")
        kwargs["explanation_style"] = body.explanation_style

    # 支持新增的个人信息字段
    if body.nickname is not None:
        kwargs["nickname"] = body.nickname.strip()
    if body.grade is not None:
        kwargs["grade"] = body.grade.strip()
    if body.gender is not None:
        if body.gender not in {"male", "female", "other", ""}:
            raise HTTPException(400, "gender 必须是 male / female / other")
        kwargs["gender"] = body.gender
    if body.mbti is not None:
        kwargs["mbti"] = body.mbti.strip().upper()

    if not kwargs:
        raise HTTPException(400, "至少提供一个字段")

    memory = await get_mem_service().update_preferences(**kwargs)

    # 写入 EverMemOS 长期记忆（后台，失败不影响响应）
    import asyncio as _asyncio
    try:
        from src.services.evermemos import get_evermemos_service
        evermemos = get_evermemos_service()
        _asyncio.create_task(evermemos.log_user_profile(memory.preferences.model_dump()))
    except Exception:
        pass

    return {"ok": True, "preferences": memory.preferences.model_dump()}


@router.get("/performance")
async def list_performance(
    subject: Optional[str] = Query(default=None, description="按科目过滤"),
    record_type: Optional[str] = Query(default=None, description="homework | exam"),
    limit: int = Query(default=30, ge=1, le=100, description="返回条数上限"),
):
    """
    返回成绩记录列表（倒序，最新在前）。
    """
    memory = await get_mem_service().load_memory()
    records = memory.performance_records
    if subject:
        records = [r for r in records if r.subject == subject]
    if record_type:
        records = [r for r in records if r.record_type == record_type]
    records = list(reversed(records[-limit:]))
    return [r.model_dump() for r in records]


@router.get("/performance/summary")
async def performance_summary(
    subject: Optional[str] = Query(default=None),
    record_type: Optional[str] = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    成绩趋势分析。

    返回：
    - `trend`：最近 N 条正确率序列（含类型、日期）
    - `homework_avg` / `exam_avg`：平时 vs 考试均值
    - `exam_anxiety_flag`：布尔，平时与考试差距 ≥ 15% 时为 true
    - `subject_avg`：各科近期平均正确率
    - `streak`：连续学习天数
    """
    return await get_mem_service().get_performance_summary(
        subject=subject, record_type=record_type, limit=limit
    )


@router.get("/streak")
async def get_streak():
    """返回连续学习天数信息。"""
    memory = await get_mem_service().load_memory()
    return memory.study_streak.model_dump()


@router.get("/knowledge-gaps")
async def get_knowledge_gaps(
    subject: Optional[str] = Query(default=None, description="按科目过滤"),
):
    """
    返回各科薄弱知识点列表。

    不传 subject 时返回全部科目，传 subject 时只返回该科目。
    """
    memory = await get_mem_service().load_memory()
    if subject:
        return {subject: memory.knowledge_gaps.get(subject, [])}
    return memory.knowledge_gaps


@router.get("/review-queue")
async def get_review_queue(
    limit: int = Query(default=8, ge=1, le=20),
):
    """
    优先复习科目建议（按薄弱知识点数量倒序）。
    前端可用此数据引导用户到错题本复习。
    """
    return await get_mem_service().get_review_queue(limit=limit)


@router.get("/greeting")
async def get_greeting():
    """
    生成个性化问候语。

    基于用户昵称、连续学习天数、薄弱科目和近期成绩生成一句鼓励语，
    同时返回用于展示用户画像的结构化数据。
    """
    from datetime import datetime as _dt
    memory  = await get_mem_service().load_memory()
    pref    = memory.preferences
    streak  = memory.study_streak
    gaps    = memory.knowledge_gaps

    name    = pref.nickname or "同学"
    hour    = _dt.now().hour
    time_greet = "早上好" if hour < 12 else ("下午好" if hour < 18 else "晚上好")

    # 最薄弱科目（知识点数最多）
    _CN = {"math":"数学","physics":"物理","chemistry":"化学",
           "english":"英语","chinese":"语文","history":"历史","biology":"生物"}
    weak_subj = sorted(gaps, key=lambda s: len(gaps[s]), reverse=True)[:2]
    weak_str  = "、".join(_CN.get(s, s) for s in weak_subj)

    # 近期正确率趋势
    recent_records = memory.performance_records[-5:]
    avg_acc = (
        sum(r.accuracy_rate for r in recent_records) / len(recent_records)
        if recent_records else None
    )

    # 拼接问候
    parts: list[str] = [f"{time_greet}，{name}！"]
    if streak.current_streak > 0:
        parts.append(f"你已坚持连续学习 {streak.current_streak} 天")
        if streak.current_streak >= 7:
            parts.append("，这种坚持真的很了不起！")
        else:
            parts.append("，每天进步一点点。")
    else:
        parts.append("每次使用都代表着一次学习的进步。")

    if weak_str:
        parts.append(f"今天可以重点练练{weak_str}哦。")

    if avg_acc is not None:
        if avg_acc >= 0.85:
            parts.append("近期状态很好，继续保持！")
        elif avg_acc >= 0.65:
            parts.append("近期稳步提升，加油！")
        else:
            parts.append("别灰心，基础打好了就会突飞猛进。")

    return {
        "greeting":    "".join(parts),
        "name":        name,
        "grade":       pref.grade,
        "gender":      pref.gender,
        "mbti":           pref.mbti,
        "streak":      streak.current_streak,
        "avg_accuracy": round(avg_acc, 3) if avg_acc is not None else None,
        "weak_subjects": weak_subj,
    }
