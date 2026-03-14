"""
WrongBookService — 错题本持久化服务
=====================================

职责：
    将批改结果中的错题/半对题写入本地 JSON 文件，并提供增删改查接口。

存储结构：
    data/wrong_book/entries.json   ← 所有错题条目（JSON 数组）

线程安全：
    使用 asyncio.Lock，适合单进程 FastAPI 服务。

用法::

    svc = WrongBookService(Path("data"))
    saved = await svc.save_from_result(homework_result)   # 批改后自动存档
    entries = await svc.list_entries(subject="math")      # 列出错题
    entry  = await svc.mark_reviewed("math_20260228_1")   # 标记已复习
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.homework.models import (
    GradeResult,
    HomeworkResult,
    WrongBookEntry,
)
from src.logging import get_logger

logger = get_logger("WrongBook")


class WrongBookService:
    """
    错题本 CRUD 服务（文件存储版）。

    Args:
        data_dir: 项目 data/ 目录的 Path 对象。
                  entries.json 将存放在 {data_dir}/wrong_book/entries.json。
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "wrong_book" / "entries.json"
        self._lock = asyncio.Lock()

    # ─────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────

    async def save_from_result(
        self,
        result: HomeworkResult,
        source_exam: Optional[str] = None,
        source_image_path: Optional[str] = None,
    ) -> int:
        """
        从 HomeworkResult 提取错题（wrong + partial）写入错题本。

        如果同一 entry_id 已存在，则**跳过**（避免重复提交同一份作业时重复存档）。

        Args:
            result:      完整批改结果
            source_exam: 试卷来源描述（可选，如"第三单元测验"）

        Returns:
            实际新增的条目数（已存在的跳过）
        """
        wrong_grades = {GradeResult.WRONG, GradeResult.PARTIAL}
        wrong_qs = [q for q in result.questions if q.grade in wrong_grades]

        if not wrong_qs:
            logger.info("[WrongBook] 本次作业全部正确，无错题入库")
            return 0

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        new_entries: list[WrongBookEntry] = []
        for q in wrong_qs:
            safe_num = re.sub(r"[^\w\u4e00-\u9fff]", "_", q.number)
            entry_id = f"{result.subject}_{date_str}_{safe_num}"
            new_entries.append(
                WrongBookEntry(
                    entry_id=entry_id,
                    subject=result.subject,
                    source_exam=source_exam,
                    question_text=q.question_text,
                    student_answer=q.student_answer,
                    correct_answer=q.correct_answer,
                    question_type=q.question_type,
                    grade=q.grade,
                    error_type=q.error_type,
                    brief_comment=q.brief_comment,
                    knowledge_points=q.knowledge_points,
                    difficulty=q.difficulty,
                    source_image_path=source_image_path,
                    question_number=q.number,
                )
            )

        async with self._lock:
            existing = self._read()
            existing_ids = {e.entry_id for e in existing}

            # 先在 new_entries 内部去重（同一次批改中可能产生相同 entry_id）
            seen: set[str] = set()
            deduped_new: list[WrongBookEntry] = []
            for e in new_entries:
                if e.entry_id not in seen:
                    deduped_new.append(e)
                    seen.add(e.entry_id)

            to_add = [e for e in deduped_new if e.entry_id not in existing_ids]
            if to_add:
                existing.extend(to_add)
                self._write(existing)

        logger.info(f"[WrongBook] 新增 {len(to_add)} 条错题（共 {len(wrong_qs)} 道错/半对题）")
        return len(to_add)

    async def list_entries(
        self,
        subject: Optional[str] = None,
        mastered: Optional[bool] = None,
    ) -> list[WrongBookEntry]:
        """
        列出错题本条目。

        Args:
            subject:  按科目过滤（None = 不过滤）
            mastered: True = 只看已掌握；False = 只看未掌握；None = 全部
        """
        async with self._lock:
            entries = self._read()

        if subject:
            entries = [e for e in entries if e.subject == subject]
        if mastered is not None:
            entries = [e for e in entries if e.mastered == mastered]

        # 按创建时间倒序
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    async def get_entry(self, entry_id: str) -> Optional[WrongBookEntry]:
        """获取单条错题。"""
        async with self._lock:
            entries = self._read()
        return next((e for e in entries if e.entry_id == entry_id), None)

    async def mark_reviewed(self, entry_id: str) -> Optional[WrongBookEntry]:
        """
        标记为已复习：review_count +1，last_reviewed_at = 当前时间。

        Returns:
            更新后的条目；entry_id 不存在时返回 None。
        """
        async with self._lock:
            entries = self._read()
            for entry in entries:
                if entry.entry_id == entry_id:
                    entry.review_count += 1
                    entry.last_reviewed_at = datetime.now().isoformat()
                    self._write(entries)
                    logger.info(f"[WrongBook] 标记复习: {entry_id} (第{entry.review_count}次)")
                    return entry
        return None

    async def mark_mastered(self, entry_id: str, mastered: bool) -> Optional[WrongBookEntry]:
        """
        设置掌握状态。

        Returns:
            更新后的条目；entry_id 不存在时返回 None。
        """
        async with self._lock:
            entries = self._read()
            for entry in entries:
                if entry.entry_id == entry_id:
                    entry.mastered = mastered
                    self._write(entries)
                    status = "已掌握" if mastered else "未掌握"
                    logger.info(f"[WrongBook] 设置{status}: {entry_id}")
                    return entry
        return None

    async def find_entry_by_question(
        self,
        question_text: str,
        subject: str,
        student_answer: str = "",
    ) -> Optional[str]:
        """
        按题目文字 + 科目 + 学生答案查找最近一条未掌握的错题，返回 entry_id。

        匹配规则：
        - subject 完全一致
        - question_text 前 80 字匹配
        - student_answer 不为空时也匹配前 80 字（忽略空格大小写）
        返回最近创建的匹配条目，未找到返回 None。
        """
        key_q = question_text[:80].strip().lower()
        key_a = student_answer[:80].strip().lower() if student_answer else None

        async with self._lock:
            entries = self._read()

        candidates = []
        for e in entries:
            if e.subject != subject or e.mastered:
                continue
            if e.question_text[:80].strip().lower() != key_q:
                continue
            if key_a is not None and e.student_answer:
                if e.student_answer[:80].strip().lower() != key_a:
                    continue
            candidates.append(e)

        if not candidates:
            return None
        # 返回最近创建的匹配条目
        candidates.sort(key=lambda e: e.created_at, reverse=True)
        return candidates[0].entry_id

    async def delete_entry(self, entry_id: str) -> bool:
        """
        删除单条错题。

        Returns:
            True = 删除成功；False = entry_id 不存在。
        """
        async with self._lock:
            entries = self._read()
            new_entries = [e for e in entries if e.entry_id != entry_id]
            if len(new_entries) == len(entries):
                return False
            self._write(new_entries)
        logger.info(f"[WrongBook] 删除条目: {entry_id}")
        return True

    async def get_stats(self) -> dict:
        """
        返回错题本统计摘要::

            {
                "total":       总条目数,
                "mastered":    已掌握数,
                "unmastered":  未掌握数,
                "by_subject":  {科目: 数量},
                "by_difficulty": {难度: 数量},
            }
        """
        async with self._lock:
            entries = self._read()

        by_subject: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        mastered_count = 0

        for e in entries:
            by_subject[e.subject] = by_subject.get(e.subject, 0) + 1
            diff = e.difficulty or "unknown"
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
            if e.mastered:
                mastered_count += 1

        return {
            "total": len(entries),
            "mastered": mastered_count,
            "unmastered": len(entries) - mastered_count,
            "by_subject": by_subject,
            "by_difficulty": by_difficulty,
        }

    # ─────────────────────────────────────────────
    # 内部：文件读写（调用前必须持有 _lock）
    # ─────────────────────────────────────────────

    def _read(self) -> list[WrongBookEntry]:
        """从磁盘读取所有条目（不存在则返回空列表）。"""
        if not self._path.exists():
            return []
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            return [WrongBookEntry.model_validate(item) for item in raw]
        except Exception as e:
            logger.error(f"[WrongBook] 读取失败: {e}，返回空列表")
            return []

    def _write(self, entries: list[WrongBookEntry]) -> None:
        """将所有条目写入磁盘。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                [e.model_dump() for e in entries],
                f,
                ensure_ascii=False,
                indent=2,
            )


__all__ = ["WrongBookService"]
