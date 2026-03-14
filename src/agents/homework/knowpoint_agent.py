"""
KnowPoint Agent — 逐题知识点标注 + 难度评级
============================================

功能：
    对已批改的题目列表，逐题标注：
    1. knowledge_points — 涉及的知识点（2～4个）
    2. difficulty       — 难度评级（easy / medium / hard / very_hard）

    结果直接写回 GradedQuestion 对象并返回。

设计原则：
    - 全部题目一次 LLM 调用（批量，高效）
    - 对正确和错误的题都标注（知识点与对错无关）
    - 知识点聚焦 8 年级数学体系（可通过 subject 适配其他科目）

用法::

    agent = KnowPointAgent()
    annotated = await agent.process(graded_questions, subject="math")
    # annotated[i].knowledge_points → ["整式乘法", "指数法则"]
    # annotated[i].difficulty       → "easy"
"""

from __future__ import annotations

import asyncio
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

_SYSTEM_PROMPT = """你是一位初中二年级（8年级）数学老师，精通人教版/北师大版数学教材知识体系。
你的任务是为每道题标注涉及的知识点和难度等级。

【知识点标注规则】
- 每题给出 2～4 个知识点，用简洁的教材术语表达
- 知识点要具体，不要写宽泛的大类（如不要只写"代数"，要写"整式乘法"或"完全平方公式"）
- 8年级数学常见知识点参考（仅供参考，以实际题目为准）：
  整式：整式乘法、单项式乘多项式、多项式乘多项式、乘法公式（完全平方、平方差）
  因式分解：提公因式法、公式法（完全平方式、平方差）、综合法
  分式：分式化简、分式四则运算、分式方程、增根
  方程：一元一次方程、含参数方程、方程无解/唯一解讨论
  函数：一次函数、正比例函数、反比例函数、函数图像
  几何：三角形全等、相似、勾股定理、平行四边形

【难度评级规则】
- easy      ：直接套公式或定义，一步到位，初学即会
- medium    ：需要2～3步推导，有一定思维量，是常规题
- hard      ：涉及多步推理或综合多个知识点，需要较强分析能力
- very_hard ：含参数讨论、逆向推导、综合应用，是压轴/拔高题

【输出格式】
只输出 JSON 数组，每个元素对应一道题，字段如下：
[
  {
    "number": "题号（与输入一致）",
    "knowledge_points": ["知识点1", "知识点2"],
    "difficulty": "easy|medium|hard|very_hard",
    "is_weak_area": false
  }
]
is_weak_area 说明：若该题的知识点命中【学生已知薄弱知识点】列表（语义相近即可），输出 true，否则输出 false。
不要输出任何其他文字。"""


def _build_question_list(
    questions: list[GradedQuestion],
    subject: str,
    known_weak_points: list[str] | None = None,
) -> str:
    """把题目压缩为给 LLM 看的纯文本，可选注入学生薄弱知识点。"""
    subject_name = {
        "math": "数学", "physics": "物理", "chemistry": "化学",
        "english": "英语", "chinese": "语文",
        "history": "历史", "biology": "生物",
        "politics": "政治", "geography": "地理",
    }.get(subject, subject)

    lines = [f"科目：{subject_name}，共 {len(questions)} 道题，请逐题标注知识点和难度。\n"]

    if known_weak_points:
        pts = "、".join(known_weak_points[:8])
        lines.append(f"【学生已知薄弱知识点】：{pts}\n")

    for q in questions:
        q_text = q.question_text[:120] + ("…" if len(q.question_text) > 120 else "")
        lines.append(f"[{q.number}] {q_text}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

_MAX_RETRIES = 2  # 最多额外重试次数（共 1+2=3 次调用）


class KnowPointAgent(BaseAgent):
    """
    逐题标注知识点和难度，将结果写回 GradedQuestion。

    输入：list[GradedQuestion]
    输出：同一个列表（knowledge_points / difficulty 字段已填充）
    """

    def __init__(self, **kwargs):
        super().__init__(
            module_name="knowpoint_agent",
            agent_name="knowpoint_agent",
            **kwargs,
        )

    async def process(
        self,
        questions: list[GradedQuestion],
        subject: str = "unknown",
        known_weak_points: list[str] | None = None,
    ) -> list[GradedQuestion]:
        """
        标注知识点和难度。支持自动重试（应对 Gemini 偶发空响应）。

        Args:
            questions: 已批改的题目列表
            subject:   科目

        Returns:
            同一列表，每道题的 knowledge_points / difficulty 已填写
        """
        if not questions:
            return questions

        self.logger.info(f"[KnowPoint] 开始标注，共 {len(questions)} 道题，科目={subject}")
        if known_weak_points:
            self.logger.info(f"[KnowPoint] 注入薄弱知识点 {len(known_weak_points)} 个")

        question_text = _build_question_list(questions, subject, known_weak_points)
        parsed = None

        # 使用全局默认 LLM（Gemini / Kimi 等，由配置决定）
        self.logger.info("[KnowPoint] 调用默认 LLM 进行知识点标注")
        for attempt in range(1, _MAX_RETRIES + 2):
            raw = await self.call_llm(
                user_prompt=question_text,
                system_prompt=_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
                stage="knowpoint",
            )
            if not raw.strip():
                self.logger.warning(
                    f"[KnowPoint] 第 {attempt}/{_MAX_RETRIES + 1} 次调用返回空响应"
                )
            else:
                parsed = self._extract_json_robust(raw)
                if parsed is not None:
                    break
                self.logger.warning(
                    f"[KnowPoint] 第 {attempt}/{_MAX_RETRIES + 1} 次 JSON 解析失败，"
                    f"原始响应前100字: {raw[:100]!r}"
                )
            if attempt <= _MAX_RETRIES:
                self.logger.info("[KnowPoint] 等待 2 秒后重试…")
                await asyncio.sleep(2)

        if parsed is None:
            self.logger.error("[KnowPoint] 所有尝试均失败，知识点标注跳过")
        else:
            self._apply_annotations(questions, parsed)

        self.logger.info(f"[KnowPoint] 标注完成")
        return questions

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    _VALID_DIFFICULTIES = {"easy", "medium", "hard", "very_hard"}

    # ─────────────────────────────────────────
    # JSON 提取（健壮版）
    # ─────────────────────────────────────────

    def _extract_json_robust(self, raw: str):
        """
        从 LLM 输出中健壮地提取 JSON 对象或数组。

        处理以下各种情况：
        - 干净的 JSON（直接返回）
        - markdown 代码块包裹：```json ... ```
        - 前后有多余说明文字
        - 数组 [...] 或对象 {...}

        返回已解析的 Python 对象（list / dict），失败时返回 None。
        """
        text = raw.strip()

        # ① 直接解析（最理想情况）
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # ② 用 find/rfind 定位最外层的 [...] 或 {...}
        #    这样可以跳过 ```json 前缀、结尾的 ``` 以及任何多余说明
        for start_ch, end_ch in [('[', ']'), ('{', '}')]:
            start = text.find(start_ch)
            end = text.rfind(end_ch)
            if start != -1 and end > start:
                chunk = text[start:end + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue

        return None

    def _apply_annotations(self, questions: list[GradedQuestion], parsed) -> None:
        """
        把已解析的 JSON（list 或 dict）中的 knowledge_points / difficulty
        写回对应的 GradedQuestion 对象（按 number 匹配）。

        Args:
            questions: 题目列表（原地修改）
            parsed:    _extract_json_robust 返回的 Python 对象（list / dict）
        """
        # LLM 有时会把数组包在对象里，如 {"questions": [...]} / {"items": [...]}
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                self.logger.error(f"[KnowPoint] 期望 list，得到 dict: {list(parsed.keys())}")
                return

        if not isinstance(parsed, list):
            self.logger.error(f"[KnowPoint] 期望 list，得到 {type(parsed)}")
            return

        # 建立 number → GradedQuestion 的映射（方便按 number 回写）
        q_map: dict[str, GradedQuestion] = {q.number: q for q in questions}

        for item in parsed:
            if not isinstance(item, dict):
                continue
            number = str(item.get("number", "")).strip()
            if not number or number not in q_map:
                # 尝试宽松匹配（去掉括号/空格）
                stripped = re.sub(r"[\(\)\s]", "", number)
                for key in q_map:
                    if re.sub(r"[\(\)\s]", "", key) == stripped:
                        number = key
                        break
                else:
                    self.logger.warning(f"[KnowPoint] 找不到题号 {number!r}，跳过")
                    continue

            q = q_map[number]

            kps = item.get("knowledge_points", [])
            if isinstance(kps, list):
                q.knowledge_points = [str(k) for k in kps if k]

            diff = str(item.get("difficulty", "")).strip().lower()
            if diff in self._VALID_DIFFICULTIES:
                q.difficulty = diff
            elif diff:
                self.logger.warning(f"[KnowPoint] 无效难度值 {diff!r}，跳过")

            if item.get("is_weak_area") is True:
                q.is_weak_area = True


__all__ = ["KnowPointAgent"]
