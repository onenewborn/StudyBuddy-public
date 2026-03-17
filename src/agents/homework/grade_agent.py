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

import asyncio
import json
import os
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

BATCH_SIZE = max(1, int(os.getenv("HOMEWORK_GRADE_BATCH_SIZE", "4")))
MAX_CONCURRENCY = max(1, int(os.getenv("HOMEWORK_GRADE_MAX_CONCURRENCY", "4")))
OBJECTIVE_BATCH_SIZE = max(
    1, int(os.getenv("HOMEWORK_GRADE_OBJECTIVE_BATCH_SIZE", str(max(BATCH_SIZE, 6))))
)
SUBJECTIVE_BATCH_SIZE = max(
    1, int(os.getenv("HOMEWORK_GRADE_SUBJECTIVE_BATCH_SIZE", str(min(BATCH_SIZE, 3))))
)
OBJECTIVE_MAX_CONCURRENCY = max(
    1, int(os.getenv("HOMEWORK_GRADE_OBJECTIVE_MAX_CONCURRENCY", str(MAX_CONCURRENCY + 1)))
)
SUBJECTIVE_MAX_CONCURRENCY = max(
    1, int(os.getenv("HOMEWORK_GRADE_SUBJECTIVE_MAX_CONCURRENCY", str(MAX_CONCURRENCY)))
)
OBJECTIVE_MAX_TOKENS = max(800, int(os.getenv("HOMEWORK_GRADE_OBJECTIVE_MAX_TOKENS", "1800")))
SUBJECTIVE_MAX_TOKENS = max(1200, int(os.getenv("HOMEWORK_GRADE_SUBJECTIVE_MAX_TOKENS", "2600")))
SPLIT_BY_TYPE = os.getenv("HOMEWORK_GRADE_SPLIT_BY_TYPE", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

_COMMON_OUTPUT_RULES = """输出要求：
1. 只输出 JSON 对象，不要有任何其他文字或 markdown 代码块。
2. 使用以下结构：
{
  "results": [
    {
      "number": "题号（与输入完全相同）",
      "correct_answer": "标准答案",
      "grade": "correct|wrong|partial|blank|skip",
      "earned_score": 0,
      "error_type": "calculation_error|concept_confusion|reading_mistake|formula_wrong|sign_error|unit_error|incomplete|logic_error|spelling_grammar|other|null",
      "brief_comment": "一句话点评，30字以内"
    }
  ]
}
3. results 中元素顺序必须与输入完全一致，数量必须相同。
4. 若学生作答为空，返回 blank；若题目不清晰无法判断，返回 skip。"""

_OBJECTIVE_SYSTEM_PROMPT = f"""你是一位严谨、专业的作业批改老师。
你正在批改客观题（选择题、填空题），请快速给出标准答案并严格比对学生答案。

判题标准：
- 选择题：correct_answer 只写选项字母或多选组合。
- 填空题：correct_answer 只写最终答案，不展开长步骤。
- 英语：大小写、常规标点不影响得分，语义正确即可判对。
- 学生答案与标准答案不一致时，一般判 wrong；只有明显接近但不完全正确时才判 partial。

{_COMMON_OUTPUT_RULES}"""

_GENERAL_SYSTEM_PROMPT = f"""你是一位严谨、专业的作业批改老师。
你需要批改学生的作业/试卷，判断每道题的对错，给出标准答案和简短点评。

判题标准：
- 选择题/填空题：答案唯一，严格对比
- 计算题/解答题：过程对但最终结果小错（如抄写失误）→ partial；思路根本错误 → wrong
- 证明题：思路正确但不完整 → partial
- 文科简答：抓住关键点即可判对，不要求逐字一致
- 英语主观题：语义对即正确，小语法问题可 partial

{_COMMON_OUTPUT_RULES}"""

_SUBJECTIVE_SYSTEM_PROMPT = f"""你是一位严谨、专业的作业批改老师。
你需要批改学生的主观题，判断每道题的对错，给出标准答案和简短点评。

判题标准：
- 计算题/解答题：过程对但最终结果小错（如抄写失误）→ partial；思路根本错误 → wrong
- 证明题：思路正确但不完整 → partial
- 文科简答：抓住关键点即可判对，不要求逐字一致
- 英语主观题：语义对即正确，小语法问题可 partial

{_COMMON_OUTPUT_RULES}"""

_USER_PROMPT_TEMPLATE = """科目：{subject}

以下是 {count} 道题目（JSON 数组），请逐题批改：
{questions_json}

请输出批改结果 JSON 对象，格式为 {{"results":[...]}}，并确保与输入顺序完全对应，共 {count} 个元素："""

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

_OBJECTIVE_TYPES = {QuestionType.CHOICE, QuestionType.FILL_BLANK}


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

        indexed_questions = list(enumerate(questions))

        if not SPLIT_BY_TYPE:
            graded_map = await self._run_partition(
                items=indexed_questions,
                subject_name=subject_name,
                memory_hint=memory_hint,
                kind="mixed",
                batch_size=BATCH_SIZE,
                max_concurrency=MAX_CONCURRENCY,
                system_prompt_base=_GENERAL_SYSTEM_PROMPT,
                max_tokens=SUBJECTIVE_MAX_TOKENS,
            )
        else:
            objective_items = [(idx, q) for idx, q in indexed_questions if q.question_type in _OBJECTIVE_TYPES]
            subjective_items = [(idx, q) for idx, q in indexed_questions if q.question_type not in _OBJECTIVE_TYPES]

            self.logger.info(
                "[Grade] 题型分流："
                f"客观题={len(objective_items)}，主观题={len(subjective_items)}"
            )

            graded_map: dict[int, GradedQuestion] = {}
            if objective_items:
                objective_results = await self._run_partition(
                    items=objective_items,
                    subject_name=subject_name,
                    memory_hint=memory_hint,
                    kind="objective",
                    batch_size=OBJECTIVE_BATCH_SIZE,
                    max_concurrency=OBJECTIVE_MAX_CONCURRENCY,
                    system_prompt_base=_OBJECTIVE_SYSTEM_PROMPT,
                    max_tokens=OBJECTIVE_MAX_TOKENS,
                )
                graded_map.update(objective_results)

            if subjective_items:
                subjective_results = await self._run_partition(
                    items=subjective_items,
                    subject_name=subject_name,
                    memory_hint=memory_hint,
                    kind="subjective",
                    batch_size=SUBJECTIVE_BATCH_SIZE,
                    max_concurrency=SUBJECTIVE_MAX_CONCURRENCY,
                    system_prompt_base=_SUBJECTIVE_SYSTEM_PROMPT,
                    max_tokens=SUBJECTIVE_MAX_TOKENS,
                )
                graded_map.update(subjective_results)

        graded_all = [graded_map[idx] for idx in range(len(questions))]

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
            _SUBJECTIVE_SYSTEM_PROMPT
            + "\n\n【申诉复审模式】这是学生对 AI 判分提出申诉的重新审核，"
            "请特别仔细独立重新评估，不要受到之前判断的影响，"
            "严格依据题目要求和学生作答客观判断。"
        )

        graded = await self._grade_batch_with_prompt(
            [fake_q],
            subject_name,
            appeal_prompt,
            max_tokens=SUBJECTIVE_MAX_TOKENS,
            stage="grade_recheck",
        )
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
        system_prompt = _GENERAL_SYSTEM_PROMPT + memory_hint
        return await self._grade_batch_with_prompt(
            batch,
            subject_name,
            system_prompt,
            max_tokens=SUBJECTIVE_MAX_TOKENS,
            stage="grade_subjective_batch",
        )

    async def _run_partition(
        self,
        items: list[tuple[int, ExtractedQuestion]],
        subject_name: str,
        memory_hint: str,
        kind: str,
        batch_size: int,
        max_concurrency: int,
        system_prompt_base: str,
        max_tokens: int,
    ) -> dict[int, GradedQuestion]:
        batch_size, max_concurrency = self._plan_partition(
            total_items=len(items),
            batch_size_cap=batch_size,
            max_concurrency_cap=max_concurrency,
        )
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        effective_concurrency = min(max_concurrency, len(batches))
        self.logger.info(
            f"[Grade] {kind} 批改：批大小={batch_size}，并发数={effective_concurrency}，共 {len(batches)} 批"
        )

        system_prompt = system_prompt_base + memory_hint
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def run_batch(
            batch_idx: int,
            batch_items: list[tuple[int, ExtractedQuestion]],
        ) -> tuple[int, list[tuple[int, GradedQuestion]]]:
            batch = [question for _, question in batch_items]
            async with semaphore:
                self.logger.info(
                    f"[Grade] {kind} 批次 {batch_idx + 1}/{len(batches)}，"
                    f"题目 {batch[0].number}~{batch[-1].number}"
                )
                graded_batch = await self._grade_batch_with_prompt(
                    batch,
                    subject_name,
                    system_prompt,
                    max_tokens=max_tokens,
                    stage=f"grade_{kind}_batch",
                )
                return batch_idx, [
                    (batch_items[idx][0], graded_batch[idx])
                    for idx in range(len(batch_items))
                ]

        batch_results = await asyncio.gather(
            *(run_batch(batch_idx, batch) for batch_idx, batch in enumerate(batches))
        )
        batch_results.sort(key=lambda item: item[0])

        graded_map: dict[int, GradedQuestion] = {}
        for _, pairs in batch_results:
            for original_idx, graded_question in pairs:
                graded_map[original_idx] = graded_question
        return graded_map

    @staticmethod
    def _plan_partition(
        *,
        total_items: int,
        batch_size_cap: int,
        max_concurrency_cap: int,
    ) -> tuple[int, int]:
        """
        动态规划批次：
        - 优先尽量 1 轮完成全部题目
        - 单批题量尽量小，降低单次调用耗时
        - 并发数不超过配置上限
        """
        if total_items <= 0:
            return 1, 1

        target_concurrency = min(max_concurrency_cap, total_items)
        planned_batch_size = max(1, (total_items + target_concurrency - 1) // target_concurrency)
        planned_batch_size = min(planned_batch_size, batch_size_cap)
        planned_batches = max(1, (total_items + planned_batch_size - 1) // planned_batch_size)
        planned_concurrency = min(max_concurrency_cap, planned_batches)
        return planned_batch_size, planned_concurrency

    async def _grade_batch_with_prompt(
        self,
        batch: list[ExtractedQuestion],
        subject_name: str,
        system_prompt: str,
        *,
        max_tokens: int,
        stage: str,
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
            max_tokens=max_tokens,
            stage=stage,
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
