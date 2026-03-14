"""
Textbook Management Router
===========================

Endpoints for uploading and managing subject textbook PDFs used by ExplainAgent RAG.

GET  /api/v1/textbook/status               — Query index status for all supported subjects
POST /api/v1/textbook/upload               — Upload a PDF for a subject, trigger background indexing
POST /api/v1/textbook/reindex/{subject}    — Force re-index (delete old + rebuild)
GET  /api/v1/textbook/file/{subject}       — Serve the PDF file for inline viewing
DELETE /api/v1/textbook/{subject}          — Delete PDF and index
PATCH /api/v1/textbook/{subject}/name      — Rename the display name of a textbook
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.agents.explain.kb_manager import TextbookKBManager, _SUBJECT_KB
from src.logging import get_logger

router = APIRouter(prefix="/api/v1/textbook", tags=["textbook"])
logger = get_logger("TextbookRouter")

_project_root = Path(__file__).resolve().parent.parent.parent.parent
_data_dir = _project_root / "data"
_kb_base = _data_dir / "knowledge_bases"
_META_FILE = _kb_base / "textbook_meta.json"   # 存储用户自定义的显示名称

_SUBJ_CN = {
    "english": "英语", "biology": "生物",
    "history": "历史", "chinese": "语文",
}


def _get_manager() -> TextbookKBManager:
    return TextbookKBManager(_data_dir)


def _load_meta() -> dict:
    """读取教材自定义元数据（显示名称等）。"""
    if _META_FILE.exists():
        try:
            return json.loads(_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_meta(meta: dict) -> None:
    _kb_base.mkdir(parents=True, exist_ok=True)
    _META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status() -> list[dict]:
    """
    返回所有已注册科目的教材索引状态。

    Response item:
        subject       科目英文名
        label         科目中文名（含用户自定义名称）
        pdf_exists    PDF 文件是否已上传
        indexed       向量库是否已就绪
        pdf_filename  PDF 文件名
        file_size_kb  文件大小（KB）
        custom_name   用户自定义显示名称（可能为空）
    """
    mgr = _get_manager()
    meta = _load_meta()
    result = []
    for subject, cfg in _SUBJECT_KB.items():
        pdf_path = _kb_base / cfg["pdf"]
        size_kb = round(pdf_path.stat().st_size / 1024) if pdf_path.exists() else 0
        result.append({
            "subject":      subject,
            "label":        _SUBJ_CN.get(subject, subject),
            "pdf_exists":   pdf_path.exists(),
            "indexed":      mgr.is_indexed(subject),
            "pdf_filename": cfg["pdf"],
            "file_size_kb": size_kb,
            "custom_name":  meta.get(subject, {}).get("name", ""),
        })
    return result


@router.post("/upload")
async def upload_textbook(
    subject: str = Form(..., description="科目：english | biology | history | chinese"),
    file: UploadFile = File(..., description="教材 PDF 文件"),
) -> dict[str, Any]:
    """
    上传教材 PDF，保存到 data/knowledge_bases/ 并在后台触发向量化。

    - 如果已有旧索引，会自动删除后重建。
    - 索引过程在后台进行（约 1-3 分钟），接口立即返回。
    """
    subject = subject.strip().lower()
    cfg = _SUBJECT_KB.get(subject)
    if not cfg:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的科目: {subject}。目前支持: {', '.join(_SUBJECT_KB)}"
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    _kb_base.mkdir(parents=True, exist_ok=True)
    pdf_dest = _kb_base / cfg["pdf"]

    # 保存 PDF
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="文件为空")
        pdf_dest.write_bytes(content)
        logger.info(f"[Textbook] 教材 PDF 已保存: {pdf_dest} ({len(content)//1024} KB)")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[Textbook] 保存 PDF 失败: {exc}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}")

    # 删除旧索引（强制重建）
    old_index_dir = _kb_base / cfg["kb_name"]
    if old_index_dir.exists():
        shutil.rmtree(old_index_dir, ignore_errors=True)
        logger.info(f"[Textbook] 已清除旧索引: {old_index_dir}")

    # 后台触发建库
    asyncio.create_task(_build_index_bg(subject, cfg))

    return {
        "ok":      True,
        "subject": subject,
        "message": f"《{cfg['desc']}》上传成功，后台向量化已启动（约 1-3 分钟）",
        "pdf":     cfg["pdf"],
    }


@router.post("/reindex/{subject}")
async def reindex_subject(subject: str) -> dict[str, Any]:
    """
    强制重新索引指定科目（要求 PDF 已存在）。
    """
    subject = subject.strip().lower()
    cfg = _SUBJECT_KB.get(subject)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"未知科目: {subject}")

    pdf_path = _kb_base / cfg["pdf"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF 不存在，请先上传: {cfg['pdf']}")

    # 删除旧索引
    old_index_dir = _kb_base / cfg["kb_name"]
    if old_index_dir.exists():
        shutil.rmtree(old_index_dir, ignore_errors=True)

    asyncio.create_task(_build_index_bg(subject, cfg))

    return {
        "ok":      True,
        "subject": subject,
        "message": f"重新索引已启动：《{cfg['desc']}》",
    }


# ── File view / delete / rename ───────────────────────────────────────────────

@router.get("/file/{subject}")
async def view_textbook_file(subject: str):
    """
    直接在浏览器内打开（或下载）指定科目的教材 PDF。
    """
    subject = subject.strip().lower()
    cfg = _SUBJECT_KB.get(subject)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"未知科目: {subject}")

    pdf_path = _kb_base / cfg["pdf"]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF 尚未上传")

    meta = _load_meta()
    custom_name = meta.get(subject, {}).get("name", "")
    display_name = custom_name or cfg["desc"]
    filename = f"{display_name}.pdf"
    filename_encoded = quote(filename, safe="")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{filename_encoded}"},
    )


@router.delete("/{subject}")
async def delete_textbook(subject: str) -> dict[str, Any]:
    """
    删除指定科目的 PDF 文件及其向量索引。
    """
    subject = subject.strip().lower()
    cfg = _SUBJECT_KB.get(subject)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"未知科目: {subject}")

    pdf_path = _kb_base / cfg["pdf"]
    index_dir = _kb_base / cfg["kb_name"]

    deleted = []
    if pdf_path.exists():
        pdf_path.unlink()
        deleted.append("PDF")
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
        deleted.append("索引")

    # 清除自定义名称
    meta = _load_meta()
    meta.pop(subject, None)
    _save_meta(meta)

    logger.info(f"[Textbook] 已删除 {subject}: {deleted}")
    return {"ok": True, "subject": subject, "deleted": deleted}


@router.patch("/{subject}/name")
async def rename_textbook(
    subject: str,
    name: str = Form(..., description="新的显示名称"),
) -> dict[str, Any]:
    """
    修改指定科目教材的显示名称（保存到 textbook_meta.json）。
    """
    subject = subject.strip().lower()
    if subject not in _SUBJECT_KB:
        raise HTTPException(status_code=404, detail=f"未知科目: {subject}")

    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")

    meta = _load_meta()
    meta.setdefault(subject, {})["name"] = name
    _save_meta(meta)

    logger.info(f"[Textbook] 重命名 {subject} → {name}")
    return {"ok": True, "subject": subject, "name": name}


# ── Background task ───────────────────────────────────────────────────────────

async def _build_index_bg(subject: str, cfg: dict) -> None:
    """后台协程：调用 TextbookKBManager 完成向量化。"""
    try:
        mgr = _get_manager()
        kb_name = await mgr.ensure_indexed(subject)
        if kb_name:
            logger.info(f"[Textbook] 后台索引完成: {subject} → {kb_name}")
        else:
            logger.error(f"[Textbook] 后台索引失败: {subject}")
    except Exception as exc:
        logger.error(f"[Textbook] 后台索引异常 ({subject}): {exc}", exc_info=True)
