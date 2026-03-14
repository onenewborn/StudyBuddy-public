"""
OCRGradeAgent — 单步 OCR + 批改合并 Agent
==========================================

将 OCRAgent 的三次 LLM 调用（vision描述 + JSON解析 + 批改）
合并为一次视觉调用，直接从图片输出已批改的结构化结果。

适用场景：
    A/B 实验 Group B，对比"分步"流水线的速度 vs 准确率差异。

核心策略：
    每张图片 → 1 次 vision LLM 调用 → list[GradedQuestion]
    （不经过中间的自由文字描述步骤）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.base_agent import BaseAgent
from src.agents.homework.models import (
    ErrorType,
    GradeResult,
    GradedQuestion,
    QuestionType,
)
from src.agents.homework.ocr_agent import OCRAgent  # 复用图片规范化逻辑

# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────

_SUBJECT_CN = {
    "math": "数学", "physics": "物理", "chemistry": "化学",
    "english": "英语", "chinese": "语文", "history": "历史", "biology": "生物",
    "politics": "政治", "geography": "地理",
}

_SUBJECT_HINTS = {
    "math":      "注意数学符号、分数、根号、方程、坐标。",
    "physics":   "注意物理公式、单位（N、m/s、kg等）、图表。",
    "chemistry": "注意化学方程式、元素符号、化学式。",
    "english":   "识别英文题目和学生的英文作答。",
    "chinese":   "注意古文、诗词、阅读题。",
    "history":   "注意人名、地名、年代。",
    "biology":   "注意生物术语、图表标注。",
    "politics":  "注意政治概念、时事材料和论述题。",
    "geography": "注意地图标注、地理术语、图表数据。",
}

_SYSTEM_TMPL = """你是一位专业的初中二年级{subject_cn}老师，同时负责识别作业图片内容和批改。
请仔细观察图片，对每道题同时完成识别和批改。
只输出 JSON 数组，不要加任何说明或 markdown 代码块。"""

_USER_TMPL = """本图是{subject_cn}作业。{hint}

请识别并批改图中每道题，输出 JSON 数组，每道题包含以下字段：
- number: 题号字符串（如"1"或"(2)"）
- question_text: 印刷题目完整文字（含选项）
- student_answer: 学生手写答案（空白则填""）
- question_type: "choice"|"fill_blank"|"calculation"|"short_answer"|"proof"|"unknown"
- score_value: 分值数字或null
- grade: "correct"|"wrong"|"partial"|"blank"|"skip"
- correct_answer: 标准答案（简洁；选择题只写字母）
- earned_score: 实际得分数字或null
- error_type: null或以下之一（仅wrong/partial时填）：
  calculation_error|concept_confusion|reading_mistake|formula_wrong|
  sign_error|unit_error|incomplete|logic_error|spelling_grammar|other
- brief_comment: 一句话点评（≤30字；correct可填"正确"）

判题规则：
- 选择/填空：严格对比，答案唯一
- 计算题：过程对但小错（如抄写）→ partial；思路根本错 → wrong
- 英语：大小写/标点不影响，语义对即 correct
- 空白作答 → blank（不要自行推断）
- 图片模糊无法识别 → grade="skip"

直接输出 JSON 数组，不要加任何说明。"""


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class OCRGradeAgent(BaseAgent):
    """
    单步 OCR + 批改合并 Agent。

    一次视觉 LLM 调用完成识别 + 批改，输出 list[GradedQuestion]。
    复用 OCRAgent 的图片压缩/规范化工具，不重复实现。
    """

    def __init__(self, **kwargs):
        super().__init__(
            module_name="ocr_grade_agent",
            agent_name="ocr_agent",   # 复用 agents.yaml 中的 ocr_agent 配置
            **kwargs,
        )
        self._normalizer = OCRAgent(**kwargs)  # 只用来调用 _normalize_image

    async def process(
        self,
        images: list[str],
        subject: str = "unknown",
    ) -> list[GradedQuestion]:
        """
        一次调用完成 OCR + 批改。

        Args:
            images:  图片列表（base64 / data URI / HTTPS URL）
            subject: 科目

        Returns:
            list[GradedQuestion] — 按页顺序排列的所有已批改题目
        """
        all_results: list[GradedQuestion] = []

        for page_idx, image in enumerate(images):
            self.logger.info(
                f"[OCRGrade] 处理第 {page_idx + 1}/{len(images)} 张图片，科目={subject}"
            )
            results = await self._process_single(image, subject, page_idx)
            all_results.extend(results)
            self.logger.info(
                f"[OCRGrade] 第 {page_idx + 1} 张：识别+批改 {len(results)} 道题"
            )

        self.logger.info(f"[OCRGrade] 共完成 {len(all_results)} 道题")
        return all_results

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    async def _process_single(
        self,
        image: str,
        subject: str,
        page_index: int,
    ) -> list[GradedQuestion]:
        """单张图片 → 1 次 vision 调用 → list[GradedQuestion]"""
        image_url = self._normalizer._normalize_image(image)
        subject_cn = _SUBJECT_CN.get(subject, subject)
        hint = _SUBJECT_HINTS.get(subject, "")

        system_prompt = _SYSTEM_TMPL.format(subject_cn=subject_cn)
        user_text = _USER_TMPL.format(subject_cn=subject_cn, hint=hint)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ],
            },
        ]

        raw = await self.call_llm(
            user_prompt="",
            system_prompt="",
            messages=messages,
            stage="ocr_grade_merged",
        )

        return self._parse(raw, subject, page_index)

    def _parse(self, raw: str, subject: str, page_index: int) -> list[GradedQuestion]:
        """解析合并输出的 JSON → list[GradedQuestion]"""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    self.logger.error(f"[OCRGrade] JSON 解析失败：{text[:300]}")
                    return []
            else:
                self.logger.error(f"[OCRGrade] 未找到 JSON：{text[:300]}")
                return []

        if isinstance(data, dict):
            data = data.get("questions", data.get("results", [data]))
        if not isinstance(data, list):
            return []

        results: list[GradedQuestion] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                grade = self._safe_grade(item.get("grade"))
                error_type = (
                    self._safe_error_type(item.get("error_type"))
                    if grade in (GradeResult.WRONG, GradeResult.PARTIAL)
                    else None
                )
                score_value = self._safe_float(item.get("score_value"))
                earned_score = self._safe_float(item.get("earned_score"))

                # 若模型未给 earned_score，按规则推算
                if earned_score is None and score_value is not None:
                    if grade == GradeResult.CORRECT:
                        earned_score = score_value
                    elif grade in (GradeResult.WRONG, GradeResult.BLANK):
                        earned_score = 0.0
                    elif grade == GradeResult.PARTIAL:
                        earned_score = round(score_value / 2, 1)

                gq = GradedQuestion(
                    number=str(item.get("number", "?")),
                    question_text=str(item.get("question_text", "")),
                    student_answer=str(item.get("student_answer", "")),
                    question_type=self._safe_qtype(item.get("question_type")),
                    score_value=score_value,
                    page_index=page_index,
                    correct_answer=str(item.get("correct_answer", "")),
                    grade=grade,
                    earned_score=earned_score,
                    error_type=error_type,
                    brief_comment=str(item.get("brief_comment", ""))[:50],
                    knowledge_points=[],
                    difficulty=None,
                )
                results.append(gq)
            except Exception as e:
                self.logger.warning(f"[OCRGrade] 跳过一道题：{e} | 数据：{item}")

        return results

    # ─────────────────────────────────────────
    # Helpers（与 OCRAgent / GradeAgent 一致）
    # ─────────────────────────────────────────

    @staticmethod
    def _safe_grade(value) -> GradeResult:
        if not value:
            return GradeResult.SKIP
        try:
            return GradeResult(str(value).lower())
        except ValueError:
            return GradeResult.SKIP

    @staticmethod
    def _safe_error_type(value) -> ErrorType | None:
        if not value:
            return None
        try:
            return ErrorType(str(value).lower())
        except ValueError:
            return ErrorType.OTHER

    @staticmethod
    def _safe_qtype(value) -> QuestionType:
        if not value:
            return QuestionType.UNKNOWN
        try:
            return QuestionType(str(value).lower())
        except ValueError:
            return QuestionType.UNKNOWN

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
