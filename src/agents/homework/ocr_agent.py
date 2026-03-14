"""
OCR Agent — 作业/试卷图片识别
==============================

功能：
    接收一张（或多张）作业/试卷图片，
    调用 Gemini 2.5 Pro 的视觉能力，
    将图中所有题目 + 学生作答 提取成结构化 JSON。

设计原则：
    - 只做识别，不做判题（判题交给 grade_agent）
    - 支持 base64 图片（上传） 和 URL 图片
    - 多图时逐张处理，page_index 记录来源
    - 输出是 list[ExtractedQuestion]，便于下游 agent 使用
    - 在 main.yaml 中配置 subject，帮助模型聚焦正确的符号体系

用法（内部）::

    agent = OCRAgent()
    questions = await agent.process(
        images=["data:image/jpeg;base64,..."],
        subject="math",
    )
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from pathlib import Path

# 保证可以从项目根目录直接运行本文件做调试
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.base_agent import BaseAgent
from src.agents.homework.models import ExtractedQuestion, QuestionType
from src.config.model_registry import agent_kwargs as _agent_kwargs

# 图片压缩阈值：base64 超过此字节数则先压缩
_MAX_B64_BYTES = 2 * 1024 * 1024   # 2 MB (base64)
_MAX_LONG_SIDE = 2048               # 压缩后长边不超过 2048px
_OCR_MAX_RETRIES = 2                # 空响应时最多重试次数


# ─────────────────────────────────────────────────────────────
# Prompt 常量
# ─────────────────────────────────────────────────────────────

# ── 第一步：让 Gemini 自由描述图片（不要求 JSON，成功率高）────────────
_DESCRIBE_SYSTEM = "你是专业的试卷内容抄录员。"

_DESCRIBE_USER_TMPL = """{subject_hint}

请从上到下逐题描述图片内容，对每道题说明：
① 题号（如"1.(1)"、"第2题"）
② 印刷题目的完整文字（含数学公式，尽量用文字符号还原，如 x²、√2、÷）
③ 学生手写的解题过程和最终答案（若空白则写"未作答"）
④ 题型（计算题/填空题/选择题/证明题/简答题）
⑤ 分值（若图中有标注）

如果某处看不清楚，写 [模糊]，不要补全猜测。
请用自然语言描述，不需要 JSON。"""

# ── 第二步：纯文字 → 结构化 JSON（无图片，稳定可靠）──────────────────
_PARSE_SYSTEM = """你是 JSON 格式化工具。
请将用户提供的试卷描述文字，整理成如下 JSON 数组，只输出 JSON，不加任何说明：
[
  {
    "number": "题号字符串，如 '(1)' 或 '1'",
    "question_text": "印刷题目原文",
    "student_answer": "学生作答（空白则为空字符串）",
    "question_type": "choice|fill_blank|calculation|short_answer|proof|unknown",
    "score_value": 分值数字或 null
  }
]
规则：
- 每个小问（(1)(2)(3)…）单独一条记录
- question_text 只写印刷体，不含学生手写内容
- 若描述中说"未作答"，student_answer 填 ""
- JSON 中不要有注释"""

_SUBJECT_HINTS = {
    "math":      "本张图片是数学作业，注意识别数学符号、分数、根号、方程、坐标等。",
    "physics":   "本张图片是物理作业，注意识别物理公式、单位（N、m/s、kg等）、图表。",
    "chemistry": "本张图片是化学作业，注意识别化学方程式、元素符号、化学式。",
    "english":   "本张图片是英语作业，识别英文题目和学生的英文作答。",
    "chinese":   "本张图片是语文作业，注意识别古文、诗词、阅读题和作文。",
    "history":   "本张图片是历史作业，注意识别人名、地名、年代等。",
    "biology":   "本张图片是生物作业，注意识别生物术语、图表标注。",
    "politics":  "本张图片是政治作业，注意识别政治概念、时事材料和论述题。",
    "geography": "本张图片是地理作业，注意识别地图标注、地理术语、图表数据。",
}

_DEFAULT_SUBJECT_HINT = "请仔细识别图片中的所有题目和学生作答。"


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

class OCRAgent(BaseAgent):
    """
    作业/试卷图片 → 结构化题目列表。

    调用 Gemini 2.5 Pro 视觉能力，通过 OpenAI-compatible messages 接口
    传入图片（base64 或 URL），输出 list[ExtractedQuestion]。
    """

    def __init__(self, parse_model_key: str | None = None, **kwargs):
        super().__init__(
            module_name="ocr_agent",
            agent_name="ocr_agent",
            **kwargs,
        )
        # Optional faster agent for Step 2 (text-only, no vision needed)
        if parse_model_key:
            parse_kw = _agent_kwargs(parse_model_key)
            if parse_kw:
                # Create a plain OCRAgent (no parse_model_key to avoid recursion)
                self._parse_agent: "OCRAgent | None" = OCRAgent(**parse_kw)
            else:
                self._parse_agent = None
        else:
            self._parse_agent = None

    # ─────────────────────────────────────────
    # Public Interface
    # ─────────────────────────────────────────

    async def process(
        self,
        images: list[str],
        subject: str = "unknown",
    ) -> list[ExtractedQuestion]:
        """
        识别图片，提取结构化题目列表。

        Args:
            images: 图片列表。每项可以是：
                    - base64 字符串（不含 data URI 前缀，会自动加）
                    - 完整 data URI，如 "data:image/jpeg;base64,..."
                    - HTTPS URL
            subject: 科目（帮助模型聚焦），如 "math", "physics"

        Returns:
            list[ExtractedQuestion] — 按页顺序排列的所有题目
        """
        self.logger.info(f"[OCR] 开始处理 {len(images)} 张图片（并行），科目={subject}")

        tasks = [
            self._process_single_image(image, subject, page_idx)
            for page_idx, image in enumerate(images)
        ]
        results_per_page = await asyncio.gather(*tasks, return_exceptions=True)

        all_questions: list[ExtractedQuestion] = []
        for page_idx, result in enumerate(results_per_page):
            if isinstance(result, Exception):
                self.logger.error(f"[OCR] 第 {page_idx + 1} 张处理失败: {result}")
                continue
            all_questions.extend(result)
            self.logger.info(f"[OCR] 第 {page_idx + 1} 张识别到 {len(result)} 道题")

        self.logger.info(f"[OCR] 共识别 {len(all_questions)} 道题")
        return all_questions

    # ─────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────

    async def _process_single_image(
        self,
        image: str,
        subject: str,
        page_index: int,
    ) -> list[ExtractedQuestion]:
        """
        两步处理单张图片：
          Step 1 — 视觉理解：Gemini 自由描述图片内容（成功率高，无 JSON 压力）
          Step 2 — 结构提取：纯文字转 JSON（无图片，稳定快速）
        """
        # ── Step 1：视觉描述 ─────────────────────────────────────────────
        image_url = self._normalize_image(image)
        subject_hint = _SUBJECT_HINTS.get(subject, _DEFAULT_SUBJECT_HINT)
        describe_user = _DESCRIBE_USER_TMPL.format(subject_hint=subject_hint)

        vision_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": describe_user},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        description = ""
        for attempt in range(1 + _OCR_MAX_RETRIES):
            if attempt > 0:
                self.logger.warning(f"[OCR] Step1 空响应，第 {attempt} 次重试…")
            description = await self.call_llm(
                user_prompt=describe_user,
                system_prompt=_DESCRIBE_SYSTEM,
                messages=vision_messages,
                stage="ocr_describe",
            )
            if description.strip():
                break

        if not description.strip():
            self.logger.error("[OCR] Step1 视觉描述失败，跳过本图")
            return []

        self.logger.debug(f"[OCR] Step1 描述完成，{len(description)} 字")

        # ── Step 2：描述文字 → 结构化 JSON ──────────────────────────────
        # 如果配置了更快的解析 agent（纯文字模型），用它做 Step 2
        parse_caller = self._parse_agent if self._parse_agent is not None else self
        raw_json = await parse_caller.call_llm(
            user_prompt=description,
            system_prompt=_PARSE_SYSTEM,
            response_format={"type": "json_object"},
            stage="ocr_parse",
        )

        return self._parse_response(raw_json, page_index)

    def _normalize_image(self, image: str) -> str:
        """
        统一图片格式为可以放进 image_url.url 的字符串。

        - 如果已经是 https URL → 直接返回
        - 如果是 base64（纯或带 data URI 前缀）→ 检测大小，必要时先压缩
        """
        if image.startswith("http"):
            return image

        # 解析 data URI 前缀
        if image.startswith("data:"):
            # data:image/jpeg;base64,<data>
            header, b64_data = image.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
        else:
            b64_data = image
            prefix = image[:8]
            mime = "image/png" if prefix.startswith("iVBOR") else "image/jpeg"

        # 超过阈值时压缩
        if len(b64_data) > _MAX_B64_BYTES:
            b64_data, mime = self._compress_image(b64_data, mime)
            self.logger.debug(
                f"[OCR] 图片已压缩，新大小 {len(b64_data)//1024}KB (base64)"
            )

        return f"data:{mime};base64,{b64_data}"

    @staticmethod
    def _compress_image(b64_data: str, mime: str) -> tuple[str, str]:
        """
        用 Pillow 将图片缩放到长边 ≤ _MAX_LONG_SIDE，并重新编码为 JPEG。
        返回新的 (base64字符串, mime类型)。
        """
        import base64
        import io
        try:
            from PIL import Image as PILImage
        except ImportError:
            # 没有 Pillow 就原样返回
            return b64_data, mime

        raw_bytes = base64.b64decode(b64_data)
        img = PILImage.open(io.BytesIO(raw_bytes))

        # 等比缩放，长边不超过 _MAX_LONG_SIDE
        w, h = img.size
        long_side = max(w, h)
        if long_side > _MAX_LONG_SIDE:
            scale = _MAX_LONG_SIDE / long_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)

        # 统一输出为 JPEG（quality=85，适合 OCR）
        out = io.BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out, format="JPEG", quality=85)
        compressed = base64.b64encode(out.getvalue()).decode()
        return compressed, "image/jpeg"

    def _parse_response(
        self,
        raw: str,
        page_index: int,
    ) -> list[ExtractedQuestion]:
        """
        解析 LLM 返回的 JSON 字符串 → list[ExtractedQuestion]。
        健壮处理：
        - LLM 可能用 ```json ... ``` 包裹
        - LLM 可能在 JSON 外加说明文字
        - 单个 dict（只有一题）也支持
        """
        # 清理 markdown 代码块
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 尝试用正则提取第一个 JSON 数组
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    self.logger.error(f"[OCR] JSON 解析失败，原始响应：{text[:300]}")
                    return []
            else:
                self.logger.error(f"[OCR] 未找到 JSON，原始响应：{text[:300]}")
                return []

        # 支持 LLM 返回单个 dict（理论上不该发生，但做防御）
        if isinstance(data, dict):
            # 可能是 {"questions": [...]} 或者直接是单道题
            if "questions" in data:
                data = data["questions"]
            else:
                data = [data]

        if not isinstance(data, list):
            self.logger.error(f"[OCR] 期望 list，得到 {type(data)}")
            return []

        questions: list[ExtractedQuestion] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                # 用 pydantic 验证并转换，宽容处理未知字段
                q = ExtractedQuestion(
                    number=str(item.get("number", "?")),
                    question_text=str(item.get("question_text", "")),
                    student_answer=str(item.get("student_answer", "")),
                    question_type=self._safe_question_type(item.get("question_type", "unknown")),
                    score_value=self._safe_float(item.get("score_value")),
                    page_index=page_index,
                )
                questions.append(q)
            except Exception as e:
                self.logger.warning(f"[OCR] 跳过一道题（解析异常）: {e} | 数据: {item}")

        return questions

    @staticmethod
    def _safe_question_type(value: str | None) -> QuestionType:
        """宽容地将字符串转为 QuestionType，无法识别时返回 UNKNOWN。"""
        if not value:
            return QuestionType.UNKNOWN
        try:
            return QuestionType(str(value).lower())
        except ValueError:
            return QuestionType.UNKNOWN

    @staticmethod
    def _safe_float(value) -> float | None:
        """安全地将值转为 float，失败返回 None。"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────
# 快速调试入口
# ─────────────────────────────────────────────────────────────

async def _debug_with_local_image(image_path: str, subject: str = "math"):
    """从本地图片文件读取并测试 OCR Agent（调试用）。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    agent = OCRAgent()
    questions = await agent.process(images=[b64], subject=subject)
    print(f"\n共识别 {len(questions)} 道题：")
    for q in questions:
        print(f"  [{q.number}] 类型={q.question_type.value} 学生答={q.student_answer[:30]!r}")
        print(f"       题目={q.question_text[:60]!r}")
    return questions


if __name__ == "__main__":
    import asyncio
    import sys as _sys

    if len(_sys.argv) < 2:
        print("用法: python ocr_agent.py <图片路径> [科目]")
        print("示例: python ocr_agent.py /tmp/hw.jpg math")
        _sys.exit(1)

    img_path = _sys.argv[1]
    subj = _sys.argv[2] if len(_sys.argv) > 2 else "math"
    asyncio.run(_debug_with_local_image(img_path, subj))
