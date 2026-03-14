"""
History API Router
==================

路由前缀：/api/v1/history

Endpoints
---------
GET  /api/v1/history/homework          — 作业批改历史列表（倒序）
GET  /api/v1/history/homework/{id}     — 单次批改详情
GET  /api/v1/history/chat              — 聊天会话列表
GET  /api/v1/history/chat/{id}         — 单次会话详情
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.logging import get_logger

logger = get_logger("History")

router = APIRouter()

_HOMEWORK_HISTORY_DIR = _project_root / "data" / "history" / "homework"
_SESSIONS_DIR = _project_root / "data" / "memory" / "sessions"

_SUBJECT_CN = {
    "math": "数学", "physics": "物理", "chemistry": "化学",
    "english": "英语", "chinese": "语文", "history": "历史", "biology": "生物",
    "politics": "政治", "geography": "地理",
}


# ─────────────────────────────────────────────────────────────────────────────
# Homework history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/homework")
async def list_homework_history(
    subject: Optional[str] = Query(default=None, description="按科目过滤"),
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    返回作业批改历史列表（最新在前）。

    每条记录包含：id、科目、批改时间、题目总数、正确率等摘要信息。
    """
    _HOMEWORK_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(_HOMEWORK_HISTORY_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if subject and data.get("subject") != subject:
                continue
            total = data.get("total_questions", 0)
            correct = data.get("correct_count", 0)
            results.append({
                "id":              f.stem,
                "subject":         data.get("subject", ""),
                "subject_cn":      _SUBJECT_CN.get(data.get("subject", ""), data.get("subject", "")),
                "graded_at":       data.get("graded_at", ""),
                "total_questions": total,
                "correct_count":   correct,
                "wrong_count":     data.get("wrong_count", 0),
                "partial_count":   data.get("partial_count", 0),
                "blank_count":     data.get("blank_count", 0),
                "accuracy_rate":   round(correct / total, 3) if total > 0 else 0.0,
                "exam_tags":       data.get("exam_tags", []),
                "weak_knowledge_points": data.get("weak_knowledge_points", []),
            })
            if len(results) >= limit:
                break
        except Exception as e:
            logger.warning(f"[History] 读取批改记录失败 {f.name}: {e}")

    return results


@router.get("/homework/{record_id}")
async def get_homework_detail(record_id: str):
    """返回单次作业批改的完整详情（含所有题目批改结果）。"""
    path = _HOMEWORK_HISTORY_DIR / f"{record_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="批改记录不存在")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[History] 读取详情失败 {record_id}: {e}")
        raise HTTPException(status_code=500, detail="读取记录失败")


# ─────────────────────────────────────────────────────────────────────────────
# Chat session history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/chat")
async def list_chat_sessions(
    subject: Optional[str] = Query(default=None, description="按科目过滤"),
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    返回聊天会话历史列表（最新在前）。

    每条记录包含：session_id、科目、最后更新时间、轮数、摘要片段。
    """
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(_SESSIONS_DIR.glob("*.json"), reverse=True)
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            subj = data.get("subject", "")
            if subject and subj != subject:
                continue
            summary = data.get("compressed_summary", "")
            messages = data.get("messages", [])
            # last user message as preview
            preview = ""
            for m in reversed(messages):
                if m.get("role") == "user" and m.get("content", "").strip():
                    preview = m["content"][:80]
                    break
            results.append({
                "session_id":   data.get("session_id", f.stem),
                "subject":      subj,
                "subject_cn":   _SUBJECT_CN.get(subj, subj),
                "turn_count":   data.get("turn_count", 0),
                "last_updated": data.get("last_updated_at", data.get("created_at", "")),
                "has_summary":  bool(summary),
                "preview":      preview or (summary[:80] if summary else ""),
            })
            if len(results) >= limit:
                break
        except Exception as e:
            logger.warning(f"[History] 读取会话记录失败 {f.name}: {e}")

    return results


@router.get("/chat/{session_id}")
async def get_chat_session(session_id: str):
    """返回单次聊天会话的完整记录（含所有消息）。"""
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="会话记录不存在")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[History] 读取会话失败 {session_id}: {e}")
        raise HTTPException(status_code=500, detail="读取会话失败")
