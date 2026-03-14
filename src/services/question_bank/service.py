"""
QuestionBankService — 题库缓存
================================

将历次批改中的题目 + 标准答案缓存到本地，
下次遇到相同题目时直接返回答案，跳过 GradeAgent LLM 调用。

策略：
  - 缓存键：题目文字标准化后的 SHA-256 前 16 位
  - 存储：data/question_bank/{subject}.json（每科一文件）
  - 缓存命中条件：
      • 选择题 / 填空题：精确匹配 → 规则判题（student_answer vs correct_answer）
      • 计算题 / 证明题 / 简答题：仅作参考，仍走 LLM（答题过程需逐步评估）
  - 写入时机：每次 GradeAgent 批改完成后，后台异步写入
  - 不覆盖：已有答案不会被覆盖（第一次写入的答案作为权威）
  - 跳过规则：grade=skip / blank 的题不写入（无有效答案）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.logging import get_logger

logger = get_logger("QuestionBank")

# 只对客观题做缓存命中判题（主观题答案不唯一，不能直接比较）
_OBJECTIVE_TYPES = {"choice", "fill_blank"}

# 最短有效题目长度（过短的题文本不缓存，避免误匹配）
_MIN_TEXT_LEN = 8


class QuestionBankService:
    """
    本地题库缓存服务（单例安全，带异步锁）。

    Usage::
        bank = QuestionBankService(data_dir)

        # 查题（在 Grade 前）
        hit = await bank.lookup(question_text, student_answer, question_type, subject)
        if hit:
            use hit["grade"], hit["correct_answer"], hit["brief_comment"]
        else:
            # 走 LLM 批改

        # 存题（Grade 完成后，后台调用）
        await bank.save_batch(graded_questions, subject)
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "question_bank"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        # 内存缓存，避免每次查询都读磁盘
        self._mem: dict[str, dict] = {}   # subject → {version, entries}

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    async def lookup(
        self,
        question_text: str,
        student_answer: str,
        question_type: str,
        subject: str,
    ) -> Optional[dict]:
        """
        查询题库。若命中且为客观题，返回规则判题结果；否则返回 None。

        Returns:
            None — 未命中，需走 LLM
            dict  — 命中，包含 {grade, correct_answer, brief_comment, from_cache=True}
        """
        norm = self._normalize(question_text)
        if len(norm) < _MIN_TEXT_LEN:
            return None
        if question_type not in _OBJECTIVE_TYPES:
            return None   # 主观题：答案不唯一，不用缓存判题

        key = self._make_key(norm)

        async with self._lock:
            data = self._load_subject(subject)
            entry = data["entries"].get(key)

        if not entry:
            return None

        # 找到缓存 → 规则判题
        correct = entry.get("correct_answer", "").strip()
        if not correct:
            return None

        student = student_answer.strip()
        grade, comment = self._rule_grade(student, correct, question_type)

        # 异步更新命中计数（fire-and-forget）
        asyncio.create_task(self._increment_hit(subject, key))

        logger.debug(
            f"[QuestionBank] 命中: subject={subject} type={question_type} "
            f"grade={grade} key={key}"
        )
        return {
            "grade":          grade,
            "correct_answer": correct,
            "brief_comment":  comment,
            "earned_score":   None,   # 由调用方根据 score_value 推算
            "error_type":     None if grade == "correct" else "other",
            "from_cache":     True,
        }

    async def save_batch(
        self,
        graded: list,      # list[GradedQuestion]
        subject: str,
    ) -> int:
        """
        批量写入批改结果到题库。跳过 skip/blank 题（无有效答案）。

        Returns:
            写入的新条目数量
        """
        _SKIP_GRADES = {"skip", "blank"}
        saved = 0

        async with self._lock:
            data = self._load_subject(subject)
            changed = False

            for q in graded:
                grade_val = q.grade.value if hasattr(q.grade, "value") else str(q.grade)
                if grade_val in _SKIP_GRADES:
                    continue

                correct = (q.correct_answer or "").strip()
                text = (q.question_text or "").strip()
                if not correct or not text:
                    continue

                norm = self._normalize(text)
                if len(norm) < _MIN_TEXT_LEN:
                    continue

                key = self._make_key(norm)
                if key in data["entries"]:
                    continue   # 不覆盖已有答案

                qtype = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
                data["entries"][key] = {
                    "question_text":  text[:500],
                    "subject":        subject,
                    "question_type":  qtype,
                    "correct_answer": correct[:300],
                    "source_grade":   grade_val,
                    "hit_count":      0,
                    "created_at":     datetime.now().isoformat(),
                }
                changed = True
                saved += 1

            if changed:
                self._save_subject(subject, data)

        if saved:
            logger.info(f"[QuestionBank] 写入 {saved} 条新题（subject={subject}）")
        return saved

    def stats(self, subject: str) -> dict:
        """返回题库统计（题目数、总命中次数）。"""
        data = self._load_subject(subject)
        entries = data["entries"]
        return {
            "subject":    subject,
            "count":      len(entries),
            "total_hits": sum(e.get("hit_count", 0) for e in entries.values()),
        }

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """标准化题目文本用于哈希匹配：去空白、小写、保留汉字和字母数字。"""
        text = text.lower().strip()
        text = re.sub(r"\s+", "", text)              # 去所有空白
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)  # 去标点（保留字母数字汉字）
        return text

    @staticmethod
    def _make_key(normalized: str) -> str:
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _rule_grade(student: str, correct: str, question_type: str) -> tuple[str, str]:
        """
        规则判题（仅限客观题）。

        Returns:
            (grade_str, brief_comment)
        """
        if not student:
            return "blank", "未作答"

        # 选择题：直接比较字母（忽略大小写/空格）
        if question_type == "choice":
            s = re.sub(r"\s+", "", student).upper()
            c = re.sub(r"\s+", "", correct).upper()
            if s == c:
                return "correct", "正确"
            return "wrong", f"正确答案为 {correct}"

        # 填空题：去空白后比较（支持数字、简单文字）
        s = re.sub(r"\s+", "", student)
        c = re.sub(r"\s+", "", correct)
        if s.lower() == c.lower():
            return "correct", "正确"
        return "wrong", f"正确答案为 {correct}"

    def _load_subject(self, subject: str) -> dict:
        """从内存或磁盘加载题库（调用前须持有锁）。"""
        if subject not in self._mem:
            path = self._dir / f"{subject}.json"
            if path.exists():
                try:
                    self._mem[subject] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    self._mem[subject] = {"version": 1, "entries": {}}
            else:
                self._mem[subject] = {"version": 1, "entries": {}}
        return self._mem[subject]

    def _save_subject(self, subject: str, data: dict) -> None:
        """写入磁盘（调用前须持有锁）。"""
        path = self._dir / f"{subject}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def _increment_hit(self, subject: str, key: str) -> None:
        async with self._lock:
            data = self._load_subject(subject)
            if key in data["entries"]:
                data["entries"][key]["hit_count"] = (
                    data["entries"][key].get("hit_count", 0) + 1
                )
                self._save_subject(subject, data)
