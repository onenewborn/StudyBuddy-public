"""
LlamaParse OCR Agent — 使用 Llama Cloud 进行作业 OCR
====================================================

功能：
    1. 使用 LlamaParse 解析图片，获得较干净的 text/markdown
    2. 复用现有 OCRAgent 的纯文本 → 结构化 JSON 解析逻辑
    3. 输出 list[ExtractedQuestion]，与现有作业批改链路兼容
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from llama_cloud import AsyncLlamaCloud

from src.agents.homework.models import ExtractedQuestion
from src.agents.homework.ocr_agent import (
    _DEFAULT_SUBJECT_HINT,
    _SUBJECT_HINTS,
    OCRAgent,
)
from src.logging import get_logger

_DEFAULT_TIER = "cost_effective"
_DEFAULT_VERSION = "latest"
_DEFAULT_LANGUAGES = ["ch_sim", "en"]
_DEFAULT_RULE_SUBJECTS = {"math"}
_STRUCTURE_MAX_TOKENS = 2200
_STRUCTURE_TEMPERATURE = 0.2
_STRUCTURE_SYSTEM = """你是作业题目结构化提取器。请把输入文本整理成 JSON 对象，只输出 JSON：
{
  "questions": [
    {
      "number": "题号，如 1、7(2)、(1)",
      "question_text": "题目原文，包含选项但不包含学生手写过程",
      "student_answer": "学生最终答案或手写过程摘要，未作答则为空字符串",
      "question_type": "choice|fill_blank|calculation|short_answer|proof|unknown",
      "score_value": null
    }
  ]
}
规则：
- 如果题目中有 A/B/C/D 选项，放进 question_text
- 如果题目下方是学生的演算过程，放进 student_answer
- 小问要拆开，如 7(1)、7(2)
- 不要输出任何额外解释"""
logger = get_logger("LlamaParseOCR")


class LlamaParseOCRAgent(OCRAgent):
    """
    使用 LlamaParse 做 OCR，再用纯文本模型结构化题目。
    """

    def __init__(
        self,
        parse_model_key: str | None = None,
        tier: str = _DEFAULT_TIER,
        version: str = _DEFAULT_VERSION,
        languages: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(parse_model_key=parse_model_key, **kwargs)
        self._tier = tier
        self._version = version
        self._languages = languages or list(_DEFAULT_LANGUAGES)
        self._rule_subjects = self._load_rule_subjects()

        api_key = (
            os.getenv("LLAMA_CLOUD_API_KEY")
            or os.getenv("LLAMAPARSE_API_KEY")
            or os.getenv("LLAMA_PARSE_API_KEY")
        )
        if not api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY 未配置，无法使用 LlamaParse OCR")
        self._client = AsyncLlamaCloud(api_key=api_key)

    async def _process_single_image(
        self,
        image: str,
        subject: str,
        page_index: int,
    ) -> list[ExtractedQuestion]:
        """
        单张图片：
          Step 1 — LlamaParse 解析图片为 markdown/text
          Step 2 — 纯文字模型转结构化 JSON
        """
        parse_start = time.time()
        raw_content = await self._extract_raw_with_llamaparse(image)
        parse_duration = time.time() - parse_start

        if not raw_content.strip():
            self.logger.error("[LlamaParseOCR] LlamaParse 未返回可用文本")
            return []

        self.logger.info(
            f"[LlamaParseOCR] 第 {page_index + 1} 张图片解析完成，"
            f"耗时 {parse_duration:.2f}s，文本长度={len(raw_content)}"
        )

        if self._should_try_rule_parse(subject):
            rule_questions, rule_reason = self._try_rule_parse(raw_content, page_index)
            if rule_questions is not None:
                self.logger.info(
                    f"[LlamaParseOCR] 第 {page_index + 1} 张图片命中规则 parser，"
                    f"共 {len(rule_questions)} 题"
                )
                return rule_questions

            self.logger.info(
                f"[LlamaParseOCR] 第 {page_index + 1} 张图片规则 parser 回退：{rule_reason}"
            )
        else:
            self.logger.info(
                f"[LlamaParseOCR] 第 {page_index + 1} 张图片跳过规则 parser，"
                f"科目 {subject} 不在白名单 {sorted(self._rule_subjects)}"
            )

        extracted_text = self._build_parse_prompt_text(raw_content, subject)
        parse_caller = self._parse_agent if self._parse_agent is not None else self
        json_start = time.time()
        raw_json = await parse_caller.call_llm(
            user_prompt=extracted_text,
            system_prompt=_STRUCTURE_SYSTEM,
            response_format={"type": "json_object"},
            max_tokens=_STRUCTURE_MAX_TOKENS,
            temperature=_STRUCTURE_TEMPERATURE,
            stage="llamaparse_ocr_parse",
        )
        json_duration = time.time() - json_start
        self.logger.info(
            f"[LlamaParseOCR] 第 {page_index + 1} 张图片结构化完成，"
            f"耗时 {json_duration:.2f}s"
        )

        return self._parse_response(raw_json, page_index)

    async def _extract_raw_with_llamaparse(self, image: str) -> str:
        suffix, image_bytes = self._decode_image(image)
        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            file_obj = await self._client.files.create(file=temp_path, purpose="parse")
            result = await self._client.parsing.parse(
                file_id=file_obj.id,
                tier=self._tier,
                version=self._version,
                expand=["markdown_full", "text_full"],
                processing_options={
                    "ocr_parameters": {"languages": self._languages},
                },
            )

            markdown = getattr(result, "markdown_full", None) or ""
            text = getattr(result, "text_full", None) or ""
            content = markdown or text
            return content or ""
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    async def _extract_with_llamaparse(self, image: str, subject: str) -> str:
        raw = await self._extract_raw_with_llamaparse(image)
        return self._build_parse_prompt_text(raw, subject)

    @staticmethod
    def _load_rule_subjects() -> set[str]:
        raw = os.getenv("HOMEWORK_OCR_RULE_SUBJECTS", "").strip().lower()
        if not raw:
            return set(_DEFAULT_RULE_SUBJECTS)
        subjects = {item.strip() for item in raw.split(",") if item.strip()}
        return subjects or set(_DEFAULT_RULE_SUBJECTS)

    def _should_try_rule_parse(self, subject: str) -> bool:
        return subject.strip().lower() in self._rule_subjects

    @staticmethod
    def _build_parse_prompt_text(raw_content: str, subject: str) -> str:
        subject_hint = _SUBJECT_HINTS.get(subject, _DEFAULT_SUBJECT_HINT)
        return f"{subject_hint}\n\n以下是试卷 OCR 结果，请按题目结构解析：\n\n{raw_content}"

    @staticmethod
    def _decode_image(image: str) -> tuple[str, bytes]:
        if image.startswith("data:"):
            header, b64_data = image.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
        else:
            b64_data = image
            prefix = image[:8]
            mime = "image/png" if prefix.startswith("iVBOR") else "image/jpeg"

        suffix = {
            "image/png": ".png",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")

        import base64

        return suffix, base64.b64decode(b64_data)

    def _try_rule_parse(
        self,
        raw_content: str,
        page_index: int,
    ) -> tuple[list[ExtractedQuestion] | None, str]:
        parsed = self._rule_parse(raw_content)
        ok, reason = self._validate_rule_parse(parsed, raw_content)
        if not ok:
            return None, reason

        questions: list[ExtractedQuestion] = []
        for item in parsed:
            try:
                questions.append(
                    ExtractedQuestion(
                        number=item["number"],
                        question_text=item["question_text"],
                        student_answer=item.get("student_answer", ""),
                        question_type=self._safe_question_type(item.get("question_type", "unknown")),
                        score_value=None,
                        page_index=page_index,
                    )
                )
            except Exception as exc:
                return None, f"规则结果构造失败: {exc}"
        return questions, "ok"

    def _validate_rule_parse(self, parsed: list[dict], raw_content: str) -> tuple[bool, str]:
        if not parsed:
            return False, "未解析出任何题目"

        empty_question = sum(1 for item in parsed if not item.get("question_text", "").strip())
        if empty_question > 0:
            return False, f"{empty_question} 道题 question_text 为空"

        duplicate_numbers: dict[str, int] = {}
        for item in parsed:
            number = item.get("number", "").strip()
            duplicate_numbers[number] = duplicate_numbers.get(number, 0) + 1
        dup_bare_sub = [
            number for number, count in duplicate_numbers.items()
            if count > 1 and re.fullmatch(r"\(\d+\)", number)
        ]
        if dup_bare_sub:
            return False, f"存在未挂靠主题号的小问重复: {dup_bare_sub[:3]}"

        top_level_count = len(re.findall(r"(?m)^\d+[\.、]", raw_content))
        if top_level_count and len(parsed) < top_level_count:
            return False, f"题目数过少: top-level={top_level_count}, parsed={len(parsed)}"

        subq_count = len(re.findall(r"(?m)^\s*\(\d+\)", raw_content))
        parsed_subq_count = sum(1 for item in parsed if re.search(r"\(\d+\)", item.get("number", "")))
        if subq_count >= 2 and parsed_subq_count == 0:
            return False, "检测到多问结构但未成功拆分小问"

        complex_questions = [
            item["number"] for item in parsed
            if len(re.findall(r"\(\d+\)", item.get("question_text", ""))) >= 2
        ]
        if complex_questions:
            return False, f"存在未拆分的复杂多问: {complex_questions[:3]}"

        unknown_ratio = sum(
            1 for item in parsed if item.get("question_type") in {"unknown", ""}
        ) / max(len(parsed), 1)
        if len(parsed) >= 6 and unknown_ratio > 0.6:
            return False, f"unknown 题型占比过高: {unknown_ratio:.0%}"

        return True, "ok"

    def _rule_parse(self, raw_content: str) -> list[dict]:
        text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", raw_content)
        text = re.sub(r"(?m)^#\s*", "", text)
        blocks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^\d+[\.、]", stripped):
                if current:
                    blocks.append("\n".join(current).strip())
                current = [stripped]
            elif current:
                current.append(line.rstrip())
        if current:
            blocks.append("\n".join(current).strip())

        results: list[dict] = []
        for block in blocks:
            results.extend(self._parse_rule_block(block))
        return [item for item in results if item]

    def _parse_rule_block(self, block: str) -> list[dict]:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            return []
        match = re.match(r"^(\d+)[\.、]\s*(.*)$", lines[0].strip())
        if not match:
            return []
        main_number = match.group(1)
        joined = "\n".join([match.group(2).strip()] + lines[1:]).strip()

        sub_matches = list(re.finditer(r"(?m)^\s*\((\d+)\)\s*", joined))
        if not sub_matches:
            item = self._build_rule_question(main_number, joined)
            return [item] if item else []

        lead_stem = joined[:sub_matches[0].start()].strip(" \n：:;；")
        results: list[dict] = []
        for idx, sub_match in enumerate(sub_matches):
            start = sub_match.start()
            end = sub_matches[idx + 1].start() if idx + 1 < len(sub_matches) else len(joined)
            sub_content = joined[start:end]
            sub_number = sub_match.group(1)
            sub_content = re.sub(r"^\(\d+\)\s*", "", sub_content).strip()
            if lead_stem:
                sub_content = f"{lead_stem}：\n{sub_content}"
            item = self._build_rule_question(f"{main_number}({sub_number})", sub_content)
            if item:
                results.append(item)
        return results

    def _build_rule_question(self, number: str, content: str) -> dict | None:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return None

        question_lines: list[str] = []
        answer_lines: list[str] = []
        in_answer = False
        for line in lines:
            if re.match(r"^[A-D]\.", line):
                question_lines.append(line)
                continue
            if line.startswith("$=") or line.startswith("=") or line.startswith("解：") or line.startswith("当"):
                in_answer = True
            if in_answer:
                answer_lines.append(line)
            else:
                question_lines.append(line)

        question_text = "\n".join(question_lines).strip()
        student_answer = "\n".join(answer_lines).strip()

        if not student_answer:
            choice_match = re.search(r"\(([A-D])\)\s*$", question_text)
            if choice_match:
                student_answer = choice_match.group(1)
            else:
                choice_before_options = re.search(r"\(([A-D])\)\s*(?:\n|$)", question_text)
                if choice_before_options and re.search(r"(?m)^A\.", question_text):
                    student_answer = choice_before_options.group(1)
                else:
                    inline_choice = re.search(r"（([A-D])）\s*$", question_text)
                    if inline_choice:
                        student_answer = inline_choice.group(1)
                    else:
                        underline_match = re.search(r"<u>(.*?)</u>", question_text)
                        if underline_match:
                            student_answer = underline_match.group(1).strip()

        return {
            "number": number,
            "question_text": question_text,
            "student_answer": student_answer,
            "question_type": self._infer_rule_question_type(question_text, student_answer),
        }

    @staticmethod
    def _infer_rule_question_type(question_text: str, student_answer: str = "") -> str:
        text = question_text.lower()
        if "a." in text and "b." in text:
            return "choice"
        if (
            "解方程" in question_text
            or "化简" in question_text
            or "计算" in question_text
            or "分解因式" in question_text
            or "因式分解" in question_text
            or "求值" in question_text
        ):
            return "calculation"
        if "____" in question_text or "值是" in question_text or "结果是" in question_text or "<u>" in question_text:
            return "fill_blank"
        if student_answer and (
            question_text.count("$") >= 2
            or "\\frac" in question_text
            or "^" in question_text
            or any(op in question_text for op in ["+", "-", "÷", "×", "="])
        ):
            return "calculation"
        return "unknown"


def build_homework_ocr_agent(model_key: str | None, **model_kw) -> OCRAgent:
    """
    优先使用 LlamaParse OCR，失败时回退旧 OCRAgent。
    """
    provider = os.getenv("HOMEWORK_OCR_PROVIDER", "llamaparse").strip().lower()
    if provider in {"legacy", "vision", "ocr_agent"}:
        return OCRAgent(parse_model_key=model_key, **model_kw)

    parse_model_key = os.getenv("HOMEWORK_OCR_PARSE_MODEL_KEY", "").strip().lower()
    if not parse_model_key:
        parse_model_key = "kimi-turbo" if (model_key or "").strip().lower().startswith("kimi") else model_key

    try:
        return LlamaParseOCRAgent(parse_model_key=parse_model_key, **model_kw)
    except Exception as exc:
        logger.warning(f"[LlamaParseOCR] 初始化失败，回退旧 OCRAgent: {exc}")
        return OCRAgent(parse_model_key=model_key, **model_kw)
