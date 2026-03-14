"""
MemoryService — 异步文件型记忆存储服务
=========================================

所有记忆数据存储在 data/memory/ 目录：
  data/memory/user_memory.json          — 用户主记忆（偏好、成绩、薄弱知识点…）
  data/memory/sessions/{session_id}.json — 聊天会话历史
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.agents.memory.profile_schema import (
    ChatMessage,
    ChatSession,
    ChatTurn,
    MemoryContext,
    PerformanceRecord,
    UserMemory,
)
from src.logging import get_logger

logger = get_logger("MemoryService")

# 会话问答轮数达到此阈值时触发压缩
COMPRESSION_THRESHOLD = 8
# 重建 LLM messages 时最多注入的历史轮数
MAX_CONTEXT_TURNS = 6


class MemoryService:
    """处理所有记忆读写操作（单例模式，通过 get_memory_service() 获取）。"""

    def __init__(self, data_dir: Path) -> None:
        self._memory_dir = data_dir / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        (self._memory_dir / "sessions").mkdir(exist_ok=True)
        (self._memory_dir / "sessions" / "images").mkdir(exist_ok=True)
        self._memory_file = self._memory_dir / "user_memory.json"
        self._lock = asyncio.Lock()

    # ── Internal sync helpers ──────────────────────────────────────────────────

    def _load_memory_sync(self) -> UserMemory:
        """同步读取（已在锁内调用）。"""
        if self._memory_file.exists():
            try:
                raw = self._memory_file.read_text(encoding="utf-8")
                return UserMemory(**json.loads(raw))
            except Exception as exc:
                logger.warning(f"[MemoryService] load failed: {exc}, returning fresh")
        return UserMemory()

    def _save_memory_sync(self, memory: UserMemory) -> None:
        memory.updated_at = datetime.now().isoformat()
        self._memory_file.write_text(
            memory.model_dump_json(indent=2), encoding="utf-8"
        )

    # ── User Memory ───────────────────────────────────────────────────────────

    async def load_memory(self) -> UserMemory:
        """异步读取用户记忆。"""
        async with self._lock:
            return self._load_memory_sync()

    async def save_memory(self, memory: UserMemory) -> None:
        """异步写入用户记忆。"""
        async with self._lock:
            self._save_memory_sync(memory)

    # ── Preferences ───────────────────────────────────────────────────────────

    async def update_preferences(self, **kwargs: str) -> UserMemory:
        """更新偏好字段，返回更新后的 UserMemory。"""
        async with self._lock:
            memory = self._load_memory_sync()
            for k, v in kwargs.items():
                if hasattr(memory.preferences, k):
                    setattr(memory.preferences, k, v)
            self._save_memory_sync(memory)
        return memory

    # ── Performance Logging ───────────────────────────────────────────────────

    async def log_performance(self, record: PerformanceRecord) -> None:
        """
        记录一次作业/考试成绩，并联动更新：
        - knowledge_gaps（薄弱知识点）
        - study_streak（连续学习天数）
        """
        async with self._lock:
            memory = self._load_memory_sync()

            # 追加成绩记录，最多保留 100 条
            memory.performance_records.append(record)
            if len(memory.performance_records) > 100:
                memory.performance_records = memory.performance_records[-100:]

            # 更新薄弱知识点
            if record.weak_knowledge_points:
                gaps = memory.knowledge_gaps.setdefault(record.subject, [])
                for pt in record.weak_knowledge_points:
                    if pt not in gaps:
                        gaps.append(pt)
                # 每科最多保留 30 个知识点
                memory.knowledge_gaps[record.subject] = gaps[-30:]

            # 更新连续学习天数
            today_str = date.today().isoformat()
            streak = memory.study_streak
            if streak.last_study_date != today_str:
                if streak.last_study_date:
                    last_date = date.fromisoformat(streak.last_study_date)
                    delta = (date.today() - last_date).days
                    if delta == 1:
                        streak.current_streak += 1
                    elif delta > 1:
                        streak.current_streak = 1
                else:
                    streak.current_streak = 1
                streak.last_study_date = today_str
                streak.total_study_days += 1
                streak.longest_streak = max(streak.longest_streak, streak.current_streak)

            self._save_memory_sync(memory)

        logger.info(
            f"[MemoryService] Logged {record.record_type} | "
            f"subject={record.subject} accuracy={record.accuracy_rate:.0%}"
        )

    async def adjust_last_record(
        self, subject: str, delta_wrong: int = -1, delta_correct: int = 1
    ) -> bool:
        """
        调整最近一条该科目成绩记录（用于申诉覆盖：将某道错题改为正确后修正统计）。

        - delta_wrong:   wrong_count 的变化量（默认 -1）
        - delta_correct: correct_count 的变化量（默认 +1）

        Returns:
            True = 找到并更新了记录；False = 没有该科目的记录。
        """
        async with self._lock:
            memory = self._load_memory_sync()
            # 找最近一条该科目记录（列表末尾）
            for rec in reversed(memory.performance_records):
                if rec.subject == subject:
                    rec.wrong_count = max(0, rec.wrong_count + delta_wrong)
                    rec.correct_count = max(0, rec.correct_count + delta_correct)
                    total = rec.total_questions or 1
                    rec.accuracy_rate = round(rec.correct_count / total, 4)
                    self._save_memory_sync(memory)
                    logger.info(
                        f"[MemoryService] adjust_last_record subject={subject} "
                        f"delta_wrong={delta_wrong} delta_correct={delta_correct}"
                    )
                    return True
        return False

    async def update_error_patterns(
        self, subject: str, error_counts: dict[str, int]
    ) -> None:
        """累加各错误类型计数。"""
        if not error_counts:
            return
        async with self._lock:
            memory = self._load_memory_sync()
            subj_err = memory.error_pattern_counts.setdefault(subject, {})
            for err_type, cnt in error_counts.items():
                subj_err[err_type] = subj_err.get(err_type, 0) + cnt
            self._save_memory_sync(memory)

    # ── Chat Sessions ─────────────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> Path:
        return self._memory_dir / "sessions" / f"{session_id}.json"

    async def load_session(self, session_id: str) -> ChatSession:
        path = self._session_path(session_id)
        if path.exists():
            try:
                return ChatSession(**json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning(f"[MemoryService] session load failed: {exc}")
        return ChatSession(session_id=session_id)

    async def save_session(self, session: ChatSession) -> None:
        session.last_updated_at = datetime.now().isoformat()
        path = self._session_path(session.session_id)
        path.write_text(session.model_dump_json(indent=2), encoding="utf-8")

    async def add_exchange(
        self,
        session_id: str,
        subject: str,
        user_msg: str,
        assistant_msg: str,
    ) -> ChatSession:
        """添加一轮问答，返回更新后的 ChatSession。"""
        session = await self.load_session(session_id)
        session.subject = subject
        now = datetime.now().isoformat()
        session.messages.append(
            ChatMessage(role="user", content=user_msg, subject=subject, timestamp=now)
        )
        session.messages.append(
            ChatMessage(role="assistant", content=assistant_msg, subject=subject, timestamp=now)
        )
        session.turn_count += 1
        await self.save_session(session)
        return session

    async def needs_compression(self, session_id: str) -> bool:
        """判断该会话是否需要压缩（未压缩且轮数达阈值）。"""
        session = await self.load_session(session_id)
        return session.turn_count >= COMPRESSION_THRESHOLD and not session.compressed_summary

    # ── Session Image Persistence ──────────────────────────────────────────────

    async def save_session_image(
        self, session_id: str, image_bytes: bytes, content_type: str = "image/jpeg"
    ) -> str:
        """
        保存 session 图片到磁盘，返回相对 memory_dir 的路径。
        每个 session 只保存最新一张图（同名覆盖）。
        """
        ext_map = {"image/png": "png", "image/webp": "webp", "image/gif": "gif"}
        suffix = ext_map.get(content_type, "jpg")
        rel_path = f"sessions/images/{session_id}.{suffix}"
        (self._memory_dir / rel_path).write_bytes(image_bytes)
        logger.debug(
            f"[MemoryService] Saved session image: {rel_path} ({len(image_bytes)//1024}KB)"
        )
        return rel_path

    async def load_session_image(self, rel_path: str) -> Optional[bytes]:
        """从磁盘加载 session 图片，路径为空或文件不存在时返回 None。"""
        if not rel_path:
            return None
        full_path = self._memory_dir / rel_path
        if full_path.exists():
            return full_path.read_bytes()
        logger.debug(f"[MemoryService] Session image not found: {rel_path}")
        return None

    # ── Multi-turn Context ─────────────────────────────────────────────────────

    async def record_turn(
        self,
        session_id: str,
        subject: str,
        user_text: str,
        assistant_text: str,
        has_image: bool = False,
        new_image_path: Optional[str] = None,
    ) -> ChatSession:
        """
        记录一轮问答到多轮上下文（turn_records）及传统 messages（兼容旧逻辑）。

        - has_image:      本轮用户是否上传了图片
        - new_image_path: 若上传了新图片，传入磁盘相对路径（会更新 current_image_path）

        返回更新后的 ChatSession（调用方可检查是否需要压缩）。
        """
        session = await self.load_session(session_id)
        session.subject = subject
        now = datetime.now().isoformat()

        # 传统 messages（保持 UI 展示兼容性，截断 600 字）
        session.messages.append(
            ChatMessage(role="user", content=user_text[:600], subject=subject, timestamp=now)
        )
        session.messages.append(
            ChatMessage(role="assistant", content=assistant_text[:600], subject=subject, timestamp=now)
        )
        session.turn_count += 1

        # 多轮上下文记录（assistant 截断 800 字以控制 token 用量）
        session.turn_records.append(
            ChatTurn(
                user_text=user_text,
                assistant_text=assistant_text[:800],
                has_image=has_image,
            )
        )

        # 更新最新图片路径
        if new_image_path:
            session.current_image_path = new_image_path

        await self.save_session(session)
        return session

    async def get_llm_context(
        self,
        session_id: str,
        max_turns: int = MAX_CONTEXT_TURNS,
    ) -> "tuple[list[ChatTurn], Optional[bytes], str]":
        """
        获取用于重建 LLM 多轮 messages 的上下文。

        Returns:
            tuple(recent_turn_records, image_bytes, compressed_summary)
            - recent_turn_records: 最近 max_turns 轮记录（ChatTurn 列表）
            - image_bytes:         会话最新图片字节（若 current_image_path 有效），否则 None
            - compressed_summary:  历史压缩摘要字符串（若已触发过压缩）
        """
        session = await self.load_session(session_id)
        recent_turns = session.turn_records[-max_turns:]

        image_bytes: Optional[bytes] = None
        if session.current_image_path:
            image_bytes = await self.load_session_image(session.current_image_path)

        return recent_turns, image_bytes, session.compressed_summary

    # ── Memory Context for Injection ──────────────────────────────────────────

    async def get_memory_context(
        self,
        subject: str,
        session_id: Optional[str] = None,
    ) -> MemoryContext:
        """
        构建注入 ExplainAgent 的 MemoryContext：
        包含偏好、各科正确率、当前科目薄弱知识点、常见错误、会话摘要。
        """
        memory = await self.load_memory()

        # 各科近期平均正确率
        subject_strength: dict[str, float] = {}
        by_subj: dict[str, list[float]] = {}
        for rec in memory.performance_records[-60:]:
            by_subj.setdefault(rec.subject, []).append(rec.accuracy_rate)
        for s, rates in by_subj.items():
            subject_strength[s] = round(sum(rates) / len(rates), 3)

        # 当前科目薄弱知识点（最近 8 个）
        weak_points = memory.knowledge_gaps.get(subject, [])[-8:]

        # 当前科目最常见错误类型（Top 3）
        err_counts = memory.error_pattern_counts.get(subject, {})
        common_errors = sorted(err_counts, key=lambda k: err_counts[k], reverse=True)[:3]

        # 本次会话摘要
        session_summary = ""
        if session_id:
            try:
                session = await self.load_session(session_id)
                session_summary = session.compressed_summary
            except Exception:
                pass

        # 最近 5 条成绩记录
        recent_scores: list[dict] = []
        for rec in reversed(memory.performance_records[-5:]):
            recent_scores.append(
                {
                    "subject": rec.subject,
                    "type": rec.record_type,
                    "accuracy": rec.accuracy_rate,
                    "date": rec.recorded_at[:10],
                }
            )

        return MemoryContext(
            preferences=memory.preferences,
            subject_strength=subject_strength,
            recent_weak_points=weak_points,
            common_errors=common_errors,
            session_summary=session_summary,
            recent_scores=recent_scores,
        )

    # ── Analytics helpers ─────────────────────────────────────────────────────

    async def get_performance_summary(
        self,
        subject: Optional[str] = None,
        record_type: Optional[str] = None,
        limit: int = 30,
    ) -> dict:
        """
        返回成绩趋势分析：
        - 最近 N 条记录的正确率序列
        - homework vs exam 平均对比（考试焦虑检测）
        - 各科强弱分布
        """
        memory = await self.load_memory()
        records = memory.performance_records

        # 过滤
        if subject:
            records = [r for r in records if r.subject == subject]
        if record_type:
            records = [r for r in records if r.record_type == record_type]
        records = records[-limit:]

        # 趋势序列
        trend = [
            {
                "date": r.recorded_at[:10],
                "accuracy": r.accuracy_rate,
                "type": r.record_type,
                "subject": r.subject,
                "exam_name": r.exam_name,
            }
            for r in records
        ]

        # homework vs exam 均值（考试焦虑检测）
        hw_rates = [r.accuracy_rate for r in records if r.record_type == "homework"]
        ex_rates = [r.accuracy_rate for r in records if r.record_type == "exam"]
        hw_avg = round(sum(hw_rates) / len(hw_rates), 3) if hw_rates else None
        ex_avg = round(sum(ex_rates) / len(ex_rates), 3) if ex_rates else None

        # 考试焦虑：平时好但考试明显差
        anxiety_flag = (
            hw_avg is not None
            and ex_avg is not None
            and hw_avg - ex_avg >= 0.15
        )

        # 各科正确率均值
        subject_avg: dict[str, float] = {}
        subj_map: dict[str, list[float]] = {}
        for r in memory.performance_records[-60:]:
            subj_map.setdefault(r.subject, []).append(r.accuracy_rate)
        for s, rates in subj_map.items():
            subject_avg[s] = round(sum(rates) / len(rates), 3)

        return {
            "trend": trend,
            "homework_avg": hw_avg,
            "exam_avg": ex_avg,
            "exam_anxiety_flag": anxiety_flag,
            "subject_avg": subject_avg,
            "streak": memory.study_streak.model_dump(),
        }

    async def get_review_queue(self, limit: int = 10) -> list[dict]:
        """
        基于简单间隔重复（Spaced Repetition）逻辑，
        从错题本目录读取待复习题目列表（按 review_count 升序排列）。
        这里返回薄弱知识点建议，由前端跳转到错题本复习。
        """
        memory = await self.load_memory()
        # 按科目弱点数量倒序，给出重点复习科目建议
        gap_sizes = {s: len(pts) for s, pts in memory.knowledge_gaps.items()}
        top_subjects = sorted(gap_sizes, key=lambda k: gap_sizes[k], reverse=True)[:limit]
        return [
            {"subject": s, "gap_count": gap_sizes[s], "top_gaps": memory.knowledge_gaps[s][-3:]}
            for s in top_subjects
        ]
