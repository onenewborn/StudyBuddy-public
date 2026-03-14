"""
Exam Tag Agent — 试卷特征标签分析
===================================

功能：
    分析一份已批改的试卷/作业，归纳出若干特征标签，
    帮助学生和老师快速了解这套题的考查重点和难度结构。

输出示例：
    ["因式分解为主", "分式方程重点", "计算量大", "步骤完整性要求高"]

设计原则：
    - 整体分析，一次 LLM 调用
    - 输出是纯字符串标签列表，简洁可读
    - 标签数量控制在 3～6 个
    - 同时给出"薄弱知识点"（从错题中总结）

用法::

    agent = ExamTagAgent()
    tags, weak_points = await agent.process(questions, subject="math")
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
from src.agents.homework.models import GradedQuestion


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一位经验丰富的初中数学老师，擅长分析试卷的考查结构和难点分布。
你会收到一份已批改的初中二年级（8年级）试卷的题目列表，包含每题的题目、学生作答和批改结果。

【任务】
1. 分析整套试卷的考查特征，给出 3～6 个标签（exam_tags）
2. 从学生的错题（grade 为 wrong 或 partial）中提炼薄弱知识点（weak_points），给出 1～4 个

【exam_tags 参考方向】（根据实际情况选取，不必拘泥于此）
- 题型维度：如"计算题为主"、"填空题偏多"、"主观题比重大"
- 知识点维度：如"因式分解专项"、"一次函数重点"、"分式方程强化"
- 难度维度：如"基础题为主"、"有一定难度"、"综合性强"
- 易错维度：如"符号处理要求高"、"步骤完整性要求高"、"陷阱题多"
- 运算维度：如"计算量大"、"运算步骤繁琐"

【输出格式】
只输出 JSON 对象，不加任何说明文字：
{
  "exam_tags": ["标签1", "标签2", "标签3"],
  "weak_points": ["薄弱知识点1", "薄弱知识点2"]
}

注意：
- 如果学生全部答对，weak_points 可以为空数组 []
- 标签要精准简洁（≤10字/个），避免空洞的话如"题目较难"
- 若科目不是数学，标签请适配该科目"""


def _build_exam_summary(questions: list[GradedQuestion], subject: str) -> str:
    """把题目列表压缩成给 LLM 看的简洁文本。"""
    subject_name = {
        "math": "数学", "physics": "物理", "chemistry": "化学",
        "english": "英语", "chinese": "语文", "history": "历史",
        "biology": "生物", "politics": "政治", "geography": "地理",
    }.get(subject, subject)

    lines = [f"科目：{subject_name}，共 {len(questions)} 道题\n"]

    grade_icon = {"correct": "✓", "wrong": "✗", "partial": "△", "blank": "○", "skip": "?"}
    for q in questions:
        icon = grade_icon.get(q.grade.value, "?")
        q_text = q.question_text[:80] + ("…" if len(q.question_text) > 80 else "")
        line = f"[{q.number}] {icon} {q.question_type.value} | {q_text}"
        if q.error_type:
            line += f" | 错误: {q.error_type.value}"
        lines.append(line)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class ExamTagAgent(BaseAgent):
    """
    分析试卷整体特征，输出标签列表和薄弱知识点。

    输入：list[GradedQuestion]（grade_agent 的输出）
    输出：(exam_tags, weak_points) 两个字符串列表
    """

    def __init__(self, **kwargs):
        super().__init__(
            module_name="exam_tag_agent",
            agent_name="exam_tag_agent",
            **kwargs,
        )

    async def process(
        self,
        questions: list[GradedQuestion],
        subject: str = "unknown",
    ) -> tuple[list[str], list[str]]:
        """
        分析试卷特征。

        Args:
            questions: 已批改的题目列表
            subject:   科目，如 "math"、"physics"

        Returns:
            (exam_tags, weak_points)
            - exam_tags:   试卷整体特征标签，如 ["因式分解专项", "计算量大"]
            - weak_points: 学生薄弱知识点，如 ["因式分解不彻底", "符号处理"]
        """
        if not questions:
            return [], []

        self.logger.info(f"[ExamTag] 分析试卷特征，共 {len(questions)} 道题，科目={subject}")

        exam_summary = _build_exam_summary(questions, subject)

        raw = await self.call_llm(
            user_prompt=exam_summary,
            system_prompt=_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            stage="exam_tag",
        )

        exam_tags, weak_points = self._parse_response(raw)
        self.logger.info(f"[ExamTag] 标签: {exam_tags} | 薄弱点: {weak_points}")
        return exam_tags, weak_points

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    def _parse_response(self, raw: str) -> tuple[list[str], list[str]]:
        """解析 LLM 的 JSON 响应，健壮处理 markdown 包裹等情况。"""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    self.logger.error(f"[ExamTag] JSON 解析失败: {text[:200]}")
                    return [], []
            else:
                self.logger.error(f"[ExamTag] 未找到 JSON: {text[:200]}")
                return [], []

        exam_tags   = [str(t) for t in data.get("exam_tags",  []) if t]
        weak_points = [str(w) for w in data.get("weak_points", []) if w]
        return exam_tags, weak_points


__all__ = ["ExamTagAgent"]
