"""
Homework Grading Data Models
============================

Defines the data structures flowing through the homework grading pipeline:

    Image
      ↓ ocr_agent
    list[ExtractedQuestion]
      ↓ grade_agent
    list[GradedQuestion]
      ↓ knowpoint_agent
    HomeworkResult  (with knowledge points + exam tags filled in)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class QuestionType(str, Enum):
    """题型分类"""
    CHOICE        = "choice"        # 选择题（单选/多选）
    FILL_BLANK    = "fill_blank"    # 填空题
    CALCULATION   = "calculation"  # 计算题 / 解答题（数学、物理、化学）
    SHORT_ANSWER  = "short_answer" # 简答题（文科）
    READING       = "reading"      # 阅读理解 / 完形填空
    WRITING       = "writing"      # 写作 / 作文
    PROOF         = "proof"        # 证明题
    UNKNOWN       = "unknown"      # 无法识别


class GradeResult(str, Enum):
    """单题批改结果"""
    CORRECT  = "correct"   # 完全正确 ✓
    WRONG    = "wrong"     # 完全错误 ✗
    PARTIAL  = "partial"   # 部分正确（过程对但结果错，或多选漏选等）
    BLANK    = "blank"     # 学生未作答
    SKIP     = "skip"      # 无法判断（题目图片不清晰/残缺）


class ErrorType(str, Enum):
    """错误类型（用于错题本分析和出题策略）"""
    CALCULATION_ERROR  = "calculation_error"  # 计算失误
    CONCEPT_CONFUSION  = "concept_confusion"  # 概念混淆
    READING_MISTAKE    = "reading_mistake"    # 审题错误 / 看漏条件
    FORMULA_WRONG      = "formula_wrong"      # 公式记错
    SIGN_ERROR         = "sign_error"         # 正负号错误
    UNIT_ERROR         = "unit_error"         # 单位错误
    INCOMPLETE         = "incomplete"         # 步骤不完整
    LOGIC_ERROR        = "logic_error"        # 逻辑推理错误
    SPELLING_GRAMMAR   = "spelling_grammar"   # 拼写/语法错误（英语/语文）
    OTHER              = "other"              # 其他


# ─────────────────────────────────────────────
# Core Models
# ─────────────────────────────────────────────

class ExtractedQuestion(BaseModel):
    """
    OCR Agent 输出：从图片中识别出的单道题目（含学生作答）。
    此时还未批改，correct_answer 为空。
    """
    number: str = Field(
        description="题号，如 '1', '(2)', '三', '2b'",
        examples=["1", "(2)", "三、1"],
    )
    question_text: str = Field(
        description="题目完整文字（含选项）",
    )
    student_answer: str = Field(
        default="",
        description="学生的作答（空字符串表示未作答）",
    )
    question_type: QuestionType = Field(
        default=QuestionType.UNKNOWN,
        description="题型",
    )
    score_value: Optional[float] = Field(
        default=None,
        description="本题分值（如试卷有标注，如 '(3分)'）",
    )
    page_index: int = Field(
        default=0,
        description="来自第几张图片（支持多图上传）",
    )


class GradedQuestion(BaseModel):
    """
    Grade Agent 输出：已批改的单道题目。
    继承 ExtractedQuestion 的所有字段，追加批改结果。
    """
    # ---- 来自 OCR ----
    number: str
    question_text: str
    student_answer: str = ""
    question_type: QuestionType = QuestionType.UNKNOWN
    score_value: Optional[float] = None
    page_index: int = 0

    # ---- 批改结果 ----
    correct_answer: str = Field(
        description="标准答案（尽量简洁，选择题只写选项字母）",
    )
    grade: GradeResult = Field(
        description="批改结论",
    )
    earned_score: Optional[float] = Field(
        default=None,
        description="实际得分（仅当 score_value 已知时计算）",
    )
    error_type: Optional[ErrorType] = Field(
        default=None,
        description="错误类型（仅 grade=wrong/partial 时填写）",
    )
    brief_comment: str = Field(
        default="",
        description="一句话点评（≤30字），指出错在哪、关键点是什么",
    )

    # ---- 知识点（由 knowpoint_agent 填写）----
    knowledge_points: list[str] = Field(
        default_factory=list,
        description="涉及的知识点，如 ['一元一次方程', '合并同类项']",
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="难度：easy / medium / hard / very_hard",
    )
    is_weak_area: bool = Field(
        default=False,
        description="该题知识点是否命中学生已知薄弱点（由 KnowPointAgent 根据记忆标注）",
    )


class HomeworkResult(BaseModel):
    """
    整份作业/试卷的批改结果（最终输出给前端和写入错题本）。
    """
    # ---- 基本信息 ----
    subject: str = Field(description="科目，如 'math', 'physics'")
    graded_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="批改时间（ISO 8601）",
    )
    image_count: int = Field(default=1, description="上传图片数量")

    # ---- 汇总统计 ----
    total_questions: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    partial_count: int = 0
    blank_count: int = 0
    skip_count: int = 0

    total_score: Optional[float] = Field(
        default=None,
        description="满分（仅当所有题目都有 score_value 时计算）",
    )
    earned_score: Optional[float] = Field(
        default=None,
        description="得分",
    )

    # ---- 考点/试卷特征（由 exam_tag_agent 填写）----
    exam_tags: list[str] = Field(
        default_factory=list,
        description="试卷特征标签，如 ['计算量大', '陷阱多', '注重基础']",
    )
    weak_knowledge_points: list[str] = Field(
        default_factory=list,
        description="本次作业暴露的薄弱知识点（错题涉及最多的）",
    )

    # ---- 题目明细 ----
    questions: list[GradedQuestion] = Field(default_factory=list)

    # ─── 便捷属性 ───────────────────────────────

    @property
    def accuracy_rate(self) -> float:
        """正确率（0.0 ~ 1.0）"""
        if self.total_questions == 0:
            return 0.0
        return self.correct_count / self.total_questions

    @property
    def wrong_questions(self) -> list[GradedQuestion]:
        """错题列表（wrong + partial）"""
        return [q for q in self.questions if q.grade in (GradeResult.WRONG, GradeResult.PARTIAL)]

    def compute_stats(self) -> None:
        """根据 questions 列表重新计算汇总统计（写入前调用）"""
        self.total_questions = len(self.questions)
        self.correct_count = sum(1 for q in self.questions if q.grade == GradeResult.CORRECT)
        self.wrong_count   = sum(1 for q in self.questions if q.grade == GradeResult.WRONG)
        self.partial_count = sum(1 for q in self.questions if q.grade == GradeResult.PARTIAL)
        self.blank_count   = sum(1 for q in self.questions if q.grade == GradeResult.BLANK)
        self.skip_count    = sum(1 for q in self.questions if q.grade == GradeResult.SKIP)

        # 有分值时计算得分
        if all(q.score_value is not None for q in self.questions) and self.questions:
            self.total_score  = sum(q.score_value for q in self.questions)   # type: ignore
            self.earned_score = sum(
                (q.earned_score or 0) for q in self.questions
            )


# ─────────────────────────────────────────────
# Wrong Book Entry（错题本单条记录）
# ─────────────────────────────────────────────

class WrongBookEntry(BaseModel):
    """
    写入错题本的单条记录。
    来源：HomeworkResult 中每道 wrong/partial 的题目。
    """
    entry_id: str = Field(description="唯一 ID，格式：{subject}_{date}_{number}")
    subject: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_exam: Optional[str] = Field(default=None, description="来源试卷名（如有）")

    question_text: str
    student_answer: str
    correct_answer: str
    question_type: QuestionType
    grade: GradeResult
    error_type: Optional[ErrorType] = None
    brief_comment: str = ""
    knowledge_points: list[str] = Field(default_factory=list)
    difficulty: Optional[str] = None

    # 原始作业信息（用于 explain 时传图片给 LLM）
    source_image_path: Optional[str] = None  # 批改时上传的原始图片路径
    question_number: str = ""                # 题目编号（如"1"/"第3题"）

    # 复习状态
    review_count: int = 0                  # 已复习次数
    last_reviewed_at: Optional[str] = None # 最近复习时间
    mastered: bool = False                 # 是否已掌握


__all__ = [
    "QuestionType",
    "GradeResult",
    "ErrorType",
    "ExtractedQuestion",
    "GradedQuestion",
    "HomeworkResult",
    "WrongBookEntry",
]
