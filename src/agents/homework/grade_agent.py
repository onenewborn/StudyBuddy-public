"""
Grade Agent — 作业批改
======================

功能：
    接收 OCR Agent 输出的 list[ExtractedQuestion]，
    逐题判断对错，给出标准答案、错误类型和一句话点评，
    输出 list[GradedQuestion]。

设计原则：
    - 批量发送（一次 LLM 调用批改多道题），节省 API 调用
    - 每批最多 BATCH_SIZE 道题，超出则分批
    - 使用 JSON 输出，通过 response_format 强制保证格式
    - 选择题依赖模型知识直接判断；计算题/证明题允许 partial
    - 不做知识点分析（交给 knowpoint_agent）

用法（内部）::

    grade_agent = GradeAgent()
    graded = await grade_agent.process(
        questions=extracted_questions,
        subject="math",
    )
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
    ExtractedQuestion,
    GradeResult,
    GradedQuestion,
    QuestionType,
)


# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────

BATCH_SIZE = 10  # 每次 LLM 调用最多批改几道题（太多会降低准确率）

_SYSTEM_PROMPT = """你是一位严谨、专业的初中二年级（8年级）作业批改老师。
你需要批改学生的作业/试卷，判断每道题的对错，给出标准答案和简短点评。

输出要求：
1. 只输出 JSON 数组，不要有任何其他文字或 markdown 代码块。
2. 数组中每个元素对应输入中的一道题（顺序一一对应，数量必须相同）。
3. 每道题包含以下字段：
   - number: 题号（与输入完全相同）
   - correct_answer: 标准答案（简洁明确：选择题只写字母，计算题写最终结果+关键步骤）
   - grade: 批改结论，只能是以下之一：
     "correct"（完全正确）/ "wrong"（错误）/ "partial"（部分正确）/
     "blank"（未作答）/ "skip"（无法判断，如题目不清晰）
   - earned_score: 实际得分（数字或 null，仅当输入中有 score_value 时计算；
     correct=满分，wrong=0，partial=约一半）
   - error_type: 错误类型（仅 grade=wrong 或 partial 时填写，否则填 null），只能是以下之一：
     "calculation_error"（计算失误）/ "concept_confusion"（概念混淆）/
     "reading_mistake"（审题错误）/ "formula_wrong"（公式错误）/
     "sign_error"（正负号错误）/ "unit_error"（单位错误）/
     "incomplete"（步骤不完整）/ "logic_error"（逻辑错误）/
     "spelling_grammar"（拼写语法）/ "other"
   - brief_comment: 一句话点评（≤30字），错的说错在哪，对的可以写"正确"

判题标准：
- 选择题/填空题：答案唯一，严格对比
- 计算题/解答题：过程对但最终结果小错（如抄写失误）→ partial；思路根本错误 → wrong
- 证明题：思路正确但不完整 → partial
- 英语：大小写/标点不影响得分，语义对即正确
- 若学生作答为空 → blank（不要自行判断）
- 若题目文字为"（图片不清晰，无法识别）"→ skip"""

_USER_PROMPT_TEMPLATE = """科目：{subject}

以下是 {count} 道题目（JSON 数组），请逐题批改：
{questions_json}

请输出批改结果 JSON 数组（与输入顺序完全对应，共 {count} 个元素）："""

_SUBJECT_NAMES = {
    "math":      "数学",
    "physics":   "物理",
    "chemistry": "化学",
    "english":   "英语",
    "chinese":   "语文",
    "history":   "历史",
    "biology":   "生物",
    "politics":  "政治",
    "geography": "地理",
}


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class GradeAgent(BaseAgent):
    """
    批改 list[ExtractedQuestion] → list[GradedQuestion]。
    """

    def __init__(self, **kwargs):
        super().__init__(
            module_name="grade_agent",
            agent_name="grade_agent",
            **kwargs,
        )

    # ─────────────────────────────────────────
    # Public Interface
    # ─────────────────────────────────────────

    async def process(
        self,
        questions: list[ExtractedQuestion],
        subject: str = "unknown",
        memory_hint: str = "",
    ) -> list[GradedQuestion]:
        """
        批改题目列表。

        Args:
            questions: OCR Agent 输出的提取题目列表
            subject:   科目，用于在 prompt 中提供上下文

        Returns:
            list[GradedQuestion] — 与输入顺序一一对应
        """
        if not questions:
            return []

        subject_name = _SUBJECT_NAMES.get(subject, subject)
        self.logger.info(f"[Grade] 开始批改，科目={subject_name}，共 {len(questions)} 道题")
        if memory_hint:
            self.logger.info("[Grade] 已注入学生记忆档案")

        # 分批处理（每批 BATCH_SIZE 道）
        graded_all: list[GradedQuestion] = []
        batches = [questions[i:i + BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            self.logger.info(
                f"[Grade] 批次 {batch_idx + 1}/{len(batches)}，"
                f"题目 {batch[0].number}~{batch[-1].number}"
            )
            graded_batch = await self._grade_batch(batch, subject_name, memory_hint)
            graded_all.extend(graded_batch)

        self.logger.info(
            f"[Grade] 批改完成：正确={sum(1 for q in graded_all if q.grade == GradeResult.CORRECT)}，"
            f"错误={sum(1 for q in graded_all if q.grade == GradeResult.WRONG)}，"
            f"部分={sum(1 for q in graded_all if q.grade == GradeResult.PARTIAL)}"
        )
        return graded_all

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    async def regrade_single(
        self,
        question_text: str,
        student_answer: str,
        subject: str,
        question_type: str = "unknown",
        correct_answer: str = "",
    ) -> dict:
        """
        对单道题重新评分（申诉复审模式）。

        用于学生认为 AI 判错时发起申诉，独立重新评估，不受之前判断影响。

        Returns:
            dict with keys: grade, correct_answer, brief_comment, error_type
        """
        try:
            qt = QuestionType(question_type)
        except ValueError:
            qt = QuestionType.UNKNOWN

        fake_q = ExtractedQuestion(
            number="1",
            question_text=question_text,
            student_answer=student_answer,
            question_type=qt,
        )

        subject_name = _SUBJECT_NAMES.get(subject, subject)
        appeal_prompt = (
            _SYSTEM_PROMPT
            + "\n\n【申诉复审模式】这是学生对 AI 判分提出申诉的重新审核，"
            "请特别仔细独立重新评估，不要受到之前判断的影响，"
            "严格依据题目要求和学生作答客观判断。"
        )

        graded = await self._grade_batch_with_prompt([fake_q], subject_name, appeal_prompt)
        gq = graded[0]
        return {
            "grade": gq.grade.value,
            "correct_answer": gq.correct_answer or correct_answer,
            "brief_comment": gq.brief_comment,
            "error_type": gq.error_type.value if gq.error_type else None,
        }

    async def _grade_batch(
        self,
        batch: list[ExtractedQuestion],
        subject_name: str,
        memory_hint: str = "",
    ) -> list[GradedQuestion]:
        """批改一批题目，返回对应的 GradedQuestion 列表。"""
        system_prompt = _SYSTEM_PROMPT + memory_hint
        return await self._grade_batch_with_prompt(batch, subject_name, system_prompt)

    async def _grade_batch_with_prompt(
        self,
        batch: list[ExtractedQuestion],
        subject_name: str,
        system_prompt: str,
    ) -> list[GradedQuestion]:
        """批改一批题目（使用指定 system_prompt），内部通用实现。"""

        # 构造发给 LLM 的题目 JSON（只保留批改需要的字段）
        questions_for_llm = [
            {
                "number": q.number,
                "question_text": q.question_text,
                "student_answer": q.student_answer if q.student_answer else "",
                "question_type": q.question_type.value,
                "score_value": q.score_value,
            }
            for q in batch
        ]

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            subject=subject_name,
            count=len(batch),
            questions_json=json.dumps(questions_for_llm, ensure_ascii=False, indent=2),
        )

        raw_response = await self.call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
            stage="grade_batch",
        )

        # 解析响应，合并回 GradedQuestion
        return self._parse_and_merge(raw_response, batch)

    def _parse_and_merge(
        self,
        raw: str,
        original_batch: list[ExtractedQuestion],
    ) -> list[GradedQuestion]:
        """
        解析 LLM 批改结果，与原始题目合并生成 GradedQuestion 列表。

        如果解析失败或数量不匹配，对无法处理的题目填写 skip。
        """
        # --- 清理 markdown 代码块 ---
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        parsed: list[dict] = []
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # 可能是 {"results": [...]} 或者直接是单道题
                if "results" in data:
                    data = data["results"]
                elif "questions" in data:
                    data = data["questions"]
                else:
                    # 只有一道题时模型返回了 dict
                    data = [data]
            if isinstance(data, list):
                parsed = data
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    self.logger.error(f"[Grade] JSON 解析失败，原始响应前300字：{text[:300]}")

        # --- 数量对齐 ---
        if len(parsed) != len(original_batch):
            self.logger.warning(
                f"[Grade] 批改结果数量不匹配：期望 {len(original_batch)}，"
                f"实际 {len(parsed)}，对齐处理"
            )

        results: list[GradedQuestion] = []
        for idx, orig_q in enumerate(original_batch):
            grade_data = parsed[idx] if idx < len(parsed) else {}

            # 从批改结果中提取字段
            grade = self._safe_grade(grade_data.get("grade"))
            error_type = (
                self._safe_error_type(grade_data.get("error_type"))
                if grade in (GradeResult.WRONG, GradeResult.PARTIAL)
                else None
            )

            # earned_score 计算：
            # 优先使用模型返回的值；如果没有但 score_value 已知，按规则推算
            earned_score = self._safe_float(grade_data.get("earned_score"))
            if earned_score is None and orig_q.score_value is not None:
                if grade == GradeResult.CORRECT:
                    earned_score = orig_q.score_value
                elif grade == GradeResult.WRONG or grade == GradeResult.BLANK:
                    earned_score = 0.0
                elif grade == GradeResult.PARTIAL:
                    earned_score = round(orig_q.score_value / 2, 1)

            gq = GradedQuestion(
                # 来自 OCR 的字段
                number=orig_q.number,
                question_text=orig_q.question_text,
                student_answer=orig_q.student_answer,
                question_type=orig_q.question_type,
                score_value=orig_q.score_value,
                page_index=orig_q.page_index,
                # 批改结果
                correct_answer=str(grade_data.get("correct_answer", "")),
                grade=grade,
                earned_score=earned_score,
                error_type=error_type,
                brief_comment=str(grade_data.get("brief_comment", ""))[:50],  # 截断过长
                # knowledge_points 和 difficulty 由 knowpoint_agent 填写
                knowledge_points=[],
                difficulty=None,
            )
            results.append(gq)

        return results

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _safe_grade(value: str | None) -> GradeResult:
        if not value:
            return GradeResult.SKIP
        try:
            return GradeResult(str(value).lower())
        except ValueError:
            return GradeResult.SKIP

    @staticmethod
    def _safe_error_type(value: str | None) -> ErrorType | None:
        if not value:
            return None
        try:
            return ErrorType(str(value).lower())
        except ValueError:
            return ErrorType.OTHER

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────
# 快速调试入口
# ─────────────────────────────────────────────────────────────

async def _debug():
    """用假数据测试 GradeAgent（无需图片）。"""
    from src.agents.homework.models import ExtractedQuestion, QuestionType

    fake_questions = [
        ExtractedQuestion(
            number="1",
            question_text="计算：3x + 5 = 14，解方程",
            student_answer="x = 2",
            question_type=QuestionType.CALCULATION,
            score_value=3.0,
        ),
        ExtractedQuestion(
            number="2",
            question_text="下列选项中，一元一次方程是（  ）\nA. x²+1=0  B. 2x-1=3  C. x+y=5  D. 1/x=2",
            student_answer="C",
            question_type=QuestionType.CHOICE,
            score_value=2.0,
        ),
        ExtractedQuestion(
            number="3",
            question_text="已知 a-b=5，求 3(a-b)+2 的值",
            student_answer="",
            question_type=QuestionType.CALCULATION,
            score_value=4.0,
        ),
    ]

    agent = GradeAgent()
    results = await agent.process(fake_questions, subject="math")

    print(f"\n批改结果（共 {len(results)} 道）：")
    for q in results:
        icon = {"correct": "✓", "wrong": "✗", "partial": "△", "blank": "○", "skip": "?"}.get(
            q.grade.value, "?"
        )
        print(f"  {icon} [{q.number}] {q.grade.value} | 正确答案={q.correct_answer!r}")
        print(f"     点评：{q.brief_comment}")
        if q.error_type:
            print(f"     错误类型：{q.error_type.value}")
        if q.earned_score is not None:
            print(f"     得分：{q.earned_score}/{q.score_value}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_debug())
