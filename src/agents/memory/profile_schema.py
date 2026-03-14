"""
Memory System Data Models
=========================

UserPreferences, StudyStreak, PerformanceRecord, ChatMessage,
ChatSession, UserMemory, MemoryContext
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── User Preferences ──────────────────────────────────────────────────────────

class UserPreferences(BaseModel):
    """用户偏好设置 + 个人基本信息。"""
    # ── 讲解偏好 ──────────────────────────────────────────────────
    explanation_length: str = Field(
        default="standard",
        description="解释详细程度：brief | standard | detailed",
    )
    explanation_style: str = Field(
        default="vivid",
        description="解释风格：vivid（生动形象）| objective（简洁清晰）",
    )
    # ── 个人信息 ──────────────────────────────────────────────────
    nickname: str = Field(default="", description="昵称")
    grade: str = Field(default="", description="年级学期，如 初二上册、高一下册")
    gender: str = Field(default="", description="性别：male | female | other")
    mbti: str = Field(default="", description="MBTI 性格类型，如 INTP、ENFJ")


# ── Study Streak ──────────────────────────────────────────────────────────────

class StudyStreak(BaseModel):
    """连续学习天数追踪。"""
    current_streak: int = 0
    longest_streak: int = 0
    last_study_date: Optional[str] = None   # ISO date string YYYY-MM-DD
    total_study_days: int = 0


# ── Performance Record ────────────────────────────────────────────────────────

class PerformanceRecord(BaseModel):
    """一次作业或考试的成绩记录。"""
    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
    )
    record_type: str = Field(description="homework | exam")
    subject: str
    exam_name: str = ""

    total_questions: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    partial_count: int = 0
    blank_count: int = 0

    earned_score: Optional[float] = None
    total_score: Optional[float] = None
    accuracy_rate: float = 0.0           # 0.0 – 1.0

    weak_knowledge_points: list[str] = Field(default_factory=list)
    exam_tags: list[str] = Field(default_factory=list)

    recorded_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )


# ── Chat Session ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """会话中的单条消息（历史兼容字段）。"""
    role: str          # "user" | "assistant"
    content: str
    subject: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )


class ChatTurn(BaseModel):
    """
    多轮上下文单轮记录——用于重建送给 LLM 的完整 messages 数组。

    与 ChatMessage 的区别：
    - ChatMessage 用于展示/记录（截断 600 字）
    - ChatTurn    用于 LLM 上下文重建（保留更多内容，含 has_image 标记）
    图片本体不存这里，保存在磁盘上，通过 ChatSession.current_image_path 引用。
    """
    user_text: str              # 用户文字内容
    assistant_text: str         # AI 回复（适度截断以控制 token）
    has_image: bool = False     # 该轮用户是否上传了图片


class ChatSession(BaseModel):
    """一次学习会话的聊天记录。"""
    session_id: str
    subject: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    compressed_summary: str = ""     # LLM 压缩后的历史摘要（turn_records 超阈值后生成）
    turn_count: int = 0              # 问答轮数（每轮 = 1问 + 1答）

    # ── 多轮上下文（新增）────────────────────────────────────────────────
    turn_records: list[ChatTurn] = Field(
        default_factory=list,
        description="近期轮次记录，用于重建 LLM 多轮 messages；超阈值后压缩并清空",
    )
    current_image_path: str = Field(
        default="",
        description="session 内最新图片的磁盘相对路径（相对 data/memory/）；图片持久化供后续轮次复用",
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )
    last_updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )


# ── User Memory (main document) ───────────────────────────────────────────────

class UserMemory(BaseModel):
    """用户的全量记忆文档，存储在 data/memory/user_memory.json。"""
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    performance_records: list[PerformanceRecord] = Field(default_factory=list)

    # 各科薄弱知识点列表
    knowledge_gaps: dict[str, list[str]] = Field(default_factory=dict)

    # 各科错误类型计数：subject -> error_type -> count
    error_pattern_counts: dict[str, dict[str, int]] = Field(default_factory=dict)

    study_streak: StudyStreak = Field(default_factory=StudyStreak)
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )


# ── Memory Context (for LLM injection) ───────────────────────────────────────

class MemoryContext(BaseModel):
    """注入 ExplainAgent system prompt 的记忆上下文。"""
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    subject_strength: dict[str, float] = Field(default_factory=dict)  # subject -> avg accuracy
    recent_weak_points: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)
    session_summary: str = ""
    recent_scores: list[dict] = Field(default_factory=list)

    def to_prompt_str(self) -> str:
        """生成注入 LLM system prompt 的记忆字符串。"""
        parts: list[str] = []

        # 个人信息
        pref = self.preferences
        identity_parts: list[str] = []
        if pref.nickname:
            identity_parts.append(f"学生昵称：{pref.nickname}")
        if pref.grade:
            identity_parts.append(f"年级：{pref.grade}")
        if pref.gender:
            gender_cn = {"male": "男生", "female": "女生"}.get(pref.gender, "")
            if gender_cn:
                identity_parts.append(f"性别：{gender_cn}")
        if pref.mbti:
            identity_parts.append(f"MBTI性格：{pref.mbti}")
        if identity_parts:
            parts.append("、".join(identity_parts) + "。")

        # 偏好
        length_map = {"brief": "简短精炼", "standard": "适中", "detailed": "详细深入"}
        style_map = {"vivid": "生动形象", "objective": "简洁客观"}
        length_cn = length_map.get(pref.explanation_length, "适中")
        style_cn = style_map.get(pref.explanation_style, "生动形象")
        parts.append(f"解释风格偏好：{style_cn}；详细程度：{length_cn}。")

        # 薄弱知识点
        if self.recent_weak_points:
            pts = "、".join(self.recent_weak_points[:6])
            parts.append(f"近期薄弱知识点：{pts}。请在讲解时优先夯实这些基础。")

        # 常见错误类型
        err_map = {
            "calculation_error": "计算失误",
            "concept_confusion": "概念混淆",
            "reading_mistake": "审题失误",
            "formula_wrong": "公式错误",
            "sign_error": "符号错误",
            "unit_error": "单位错误",
            "incomplete": "解答不完整",
            "logic_error": "逻辑错误",
            "spelling_grammar": "拼写/语法错误",
            "other": "其他",
        }
        if self.common_errors:
            errs = "、".join(err_map.get(e, e) for e in self.common_errors[:3])
            parts.append(f"常见错误类型：{errs}。请在解析时重点提醒。")

        # 本次会话摘要
        if self.session_summary:
            parts.append(f"本次学习摘要：{self.session_summary}")

        if not parts:
            return ""

        header = "\n\n【学生学习档案（请据此调整讲解方式）】\n"
        return header + "\n".join(f"• {p}" for p in parts) + "\n"
