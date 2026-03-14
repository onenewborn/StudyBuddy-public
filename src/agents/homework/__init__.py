"""StudyBuddy — 作业批改模块"""

from src.agents.homework.models import (
    ErrorType,
    ExtractedQuestion,
    GradeResult,
    GradedQuestion,
    HomeworkResult,
    QuestionType,
    WrongBookEntry,
)
from src.agents.homework.ocr_agent import OCRAgent
from src.agents.homework.grade_agent import GradeAgent
from src.agents.homework.exam_tag_agent import ExamTagAgent
from src.agents.homework.knowpoint_agent import KnowPointAgent
from src.agents.homework.ocr_grade_agent import OCRGradeAgent

__all__ = [
    # Models
    "QuestionType",
    "GradeResult",
    "ErrorType",
    "ExtractedQuestion",
    "GradedQuestion",
    "HomeworkResult",
    "WrongBookEntry",
    # Agents
    "OCRAgent",
    "GradeAgent",
    "ExamTagAgent",
    "KnowPointAgent",
    "OCRGradeAgent",
]
