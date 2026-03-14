"""
ExplainAgent — 难题解析智能体
==============================

支持两种使用场景
----------------
1. **错题解析模式** (mode="error_analysis")
   来源：错题本，有完整的学生答案 / 正确答案 / 错误类型上下文
   目标："学生把 enormous 填成 huge，为什么错了？正确答案的依据是什么？"

2. **主动问题询问模式** (mode="question_explain")
   来源：学生主动提问任何一道题
   目标："这道题怎么做？请给我讲解一下思路。"

两种模式都支持 RAG
------------------
理科（math / physics / chemistry）→ 纯 LLM
文科（english / biology / history / chinese）→ 教材 RAG + LLM
    首次请求自动建立向量索引，后续复用磁盘缓存

模式自动推断
------------
- 提供了 correct_answer 或 student_answer → error_analysis
- 否则 → question_explain
也可以通过 ExplainRequest.mode 字段手动指定。
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# ── Project root ──────────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.base_agent import BaseAgent
from src.agents.explain.kb_manager import TextbookKBManager
from src.logging import get_logger

logger = get_logger("ExplainAgent")

# 使用纯 LLM 的科目（理科通用知识不依赖特定教材版本）
_LLM_ONLY_SUBJECTS = {"math", "physics", "chemistry"}

_SUBJECT_CN: dict[str, str] = {
    "math":      "数学",
    "physics":   "物理",
    "chemistry": "化学",
    "english":   "英语",
    "biology":   "生物",
    "history":   "历史",
    "chinese":   "语文",
}


# ── Request / Response Models ─────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    """
    难题解析请求。

    支持两种场景：

    **场景一：错题解析**（从错题本发起）
        提供 student_answer + correct_answer，系统会着重分析"为什么错了"。

        示例：
        {
            "question_text": "The building was ___ (A) huge  (B) enormous",
            "subject": "english",
            "student_answer": "B",
            "correct_answer": "A",
            "knowledge_points": ["形容词", "词汇"],
            "error_type": "concept_confusion"
        }

    **场景二：主动问题询问**（学生主动提问任何题目）
        只需提供 question_text + subject，系统会讲解解题思路。

        示例：
        {
            "question_text": "解方程：2x + 5 = 13，求 x 的值",
            "subject": "math"
        }

    mode 字段由系统自动推断，也可手动指定。
    """
    question_text:    str                                      = Field(description="题目完整文字（含选项/条件）")
    subject:          str                                      = Field(default="math", description="科目")
    mode:             Optional[Literal["error_analysis",
                                       "question_explain"]]   = Field(default=None,  description="解析模式（留空自动推断）")

    # ── 错题上下文（error_analysis 时填写，question_explain 时可省略）──────────
    student_answer:   str                                      = Field(default="", description="学生作答")
    correct_answer:   str                                      = Field(default="", description="标准答案")
    question_type:    str                                      = Field(default="unknown", description="题型")
    knowledge_points: list[str]                               = Field(default_factory=list, description="涉及知识点")
    error_type:       Optional[str]                           = Field(default=None, description="错误类型")
    brief_comment:    str                                      = Field(default="", description="批改点评")

    @model_validator(mode="after")
    def _infer_mode(self) -> "ExplainRequest":
        """若 mode 未指定，根据是否有错题上下文自动推断。"""
        if self.mode is None:
            has_error_ctx = bool(self.student_answer or self.correct_answer or self.error_type)
            self.mode = "error_analysis" if has_error_ctx else "question_explain"
        return self


class ExplainResponse(BaseModel):
    """难题解析结果。"""
    question_text:     str            = Field(description="原题目")
    subject:           str            = Field(description="科目")
    mode:              str            = Field(description="实际使用的解析模式")
    explanation:       str            = Field(description="Markdown 格式的详细解析")
    key_concepts:      list[str]      = Field(default_factory=list,  description="核心知识点/概念标签")
    used_rag:          bool           = Field(default=False,          description="是否使用了教材检索")
    textbook_snippets: list[str]      = Field(default_factory=list,
                                             description="检索到的教材原文片段（前端展示「教材来源」用）")


# ── Agent ─────────────────────────────────────────────────────────────────────

class ExplainAgent(BaseAgent):
    """
    难题解析智能体。继承 BaseAgent 获得：
    - agents.yaml 中 explain_agent 的 temperature / max_tokens 配置
    - call_llm(user_prompt, system_prompt) 统一接口
    - 日志
    """

    def __init__(self, data_dir: Optional[Path] = None, **kwargs: Any) -> None:
        super().__init__(module_name="explain", agent_name="explain_agent", **kwargs)
        _data = data_dir or (_project_root / "data")
        self._kb_manager = TextbookKBManager(_data)

    # BaseAgent 抽象方法
    async def process(
        self, request: ExplainRequest, memory_context: str = "", **kwargs: Any
    ) -> ExplainResponse:
        return await self.explain(request, memory_context=memory_context)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def explain(
        self, request: ExplainRequest, memory_context: str = ""
    ) -> ExplainResponse:
        """根据科目 + 模式选择解析路径。memory_context 会注入 system prompt。"""
        subject = request.subject.lower()
        use_rag = (
            subject not in _LLM_ONLY_SUBJECTS
            and self._kb_manager.supports_subject(subject)
        )

        logger.info(
            f"[ExplainAgent] subject={subject}  mode={request.mode}  "
            f"rag={'yes' if use_rag else 'no'}  "
            f"memory={'yes' if memory_context else 'no'}"
        )

        if use_rag:
            return await self._rag_explain(request, memory_context=memory_context)
        else:
            return await self._direct_explain(request, memory_context=memory_context)

    # ── 纯 LLM 路径（理科 or PDF 未上传时的降级）────────────────────────────

    async def _direct_explain(
        self, request: ExplainRequest, memory_context: str = ""
    ) -> ExplainResponse:
        system_prompt, user_prompt = self._build_direct_prompts(request, memory_context)
        explanation = await self.call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage="direct_explain",
        )
        return ExplainResponse(
            question_text=request.question_text,
            subject=request.subject,
            mode=request.mode or "question_explain",
            explanation=explanation,
            key_concepts=self._build_key_concepts(request),
            used_rag=False,
            textbook_snippets=[],
        )

    # ── RAG + LLM 路径（文科 / 语言类）──────────────────────────────────────

    async def _rag_explain(
        self, request: ExplainRequest, memory_context: str = ""
    ) -> ExplainResponse:
        # Step 1: 确保索引存在（首次慢，后续秒级）
        kb_name = await self._kb_manager.ensure_indexed(request.subject)

        # Step 2: 检索教材片段
        snippets: list[str] = []
        if kb_name:
            query   = self._build_retrieval_query(request)
            logger.info(f"[ExplainAgent] RAG query: {query[:80]}")
            snippets = await self._kb_manager.search(kb_name, query, top_k=4)
            logger.info(f"[ExplainAgent] 检索到 {len(snippets)} 个片段")

        used_rag = bool(snippets)

        # Step 3: 构建 prompt（有检索结果用 RAG prompt，否则降级）
        if used_rag:
            system_prompt, user_prompt = self._build_rag_prompts(request, snippets, memory_context)
            stage = "rag_explain"
        else:
            logger.info("[ExplainAgent] RAG 无结果，降级为纯 LLM")
            system_prompt, user_prompt = self._build_direct_prompts(request, memory_context)
            stage = "direct_explain_fallback"

        explanation = await self.call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            stage=stage,
        )

        return ExplainResponse(
            question_text=request.question_text,
            subject=request.subject,
            mode=request.mode or "question_explain",
            explanation=explanation,
            key_concepts=self._build_key_concepts(request),
            used_rag=used_rag,
            textbook_snippets=snippets[:3],
        )

    # ── Prompt 构建 ───────────────────────────────────────────────────────────
    # 所有 prompt 都拆成 system_prompt + user_prompt，与 call_llm() 签名匹配

    def _build_direct_prompts(
        self, req: ExplainRequest, memory_context: str = ""
    ) -> tuple[str, str]:
        """
        理科 / 降级路径。
        - error_analysis : 分析为什么错了 + 正确解题步骤
        - question_explain : 讲解如何解这道题
        """
        subject_cn = _SUBJECT_CN.get(req.subject.lower(), req.subject)

        system_prompt = (
            f"你是一位初中二年级{subject_cn}老师，"
            "用简洁、亲切的语言帮助 13-14 岁学生理解难题。"
            "回答使用 Markdown 格式，层次清晰。"
            + memory_context
        )

        if req.mode == "error_analysis":
            kp_str    = "、".join(req.knowledge_points) if req.knowledge_points else "（未标注）"
            error_str = req.error_type or "（未标注）"
            comment   = req.brief_comment or "（无）"

            user_prompt = f"""【错题信息】
题目：{req.question_text}
学生答案：{req.student_answer or "（未作答）"}
正确答案：{req.correct_answer or "（见题目）"}
题型：{req.question_type}
涉及知识点：{kp_str}
错误类型：{error_str}
批改点评：{comment}

请按以下结构讲解：

## ✅ 正确解题过程
（展示完整推导步骤；数学/物理/化学要写出关键公式和推导，不要跳步）

## ❌ 错误原因分析
（针对「{req.error_type or "该错误"}」，指出学生具体哪步出错，为什么错）

## 💡 解题技巧 & 易错提醒
（同类题的解题规律，帮助下次不再犯同类错误）"""

        else:  # question_explain
            kp_str = "、".join(req.knowledge_points) if req.knowledge_points else ""
            kp_line = f"\n知识点提示：{kp_str}" if kp_str else ""

            user_prompt = f"""【题目】
{req.question_text}{kp_line}

请按以下结构讲解：

## 📝 题目分析
（理解题目考察的核心知识点，明确解题方向）

## ✅ 解题过程
（完整推导步骤，数学/物理/化学要写出关键公式，不要跳步）

## 💡 解题技巧 & 总结
（同类题型的解题规律和注意事项）"""

        return system_prompt, user_prompt

    def _build_rag_prompts(
        self, req: ExplainRequest, snippets: list[str], memory_context: str = ""
    ) -> tuple[str, str]:
        """
        文科 / 语言类：以教材内容为 ground truth 的 prompt。
        同样区分 error_analysis 和 question_explain 两种模式。
        """
        subject_cn    = _SUBJECT_CN.get(req.subject.lower(), req.subject)
        textbook_desc = self._kb_manager.get_textbook_desc(req.subject) or f"八年级{subject_cn}教材"

        system_prompt = (
            f"你是一位初中二年级{subject_cn}老师，"
            f"以《{textbook_desc}》为唯一权威参考，用亲切语言帮助 13-14 岁学生。"
            "回答使用 Markdown 格式，层次清晰。"
            "【重要】你只能根据下方提供的教材原文片段作答，"
            "不可引入教材以外的任何知识；"
            "教材中没有提及的内容，请直接说明'教材中未涉及'。"
            + memory_context
        )

        # 构建教材上下文块
        context_block = "\n\n---\n".join(
            f"【教材片段 {i + 1}】\n{s.strip()}"
            for i, s in enumerate(snippets)
        )
        rag_header = (
            f"以下是从《{textbook_desc}》中检索到的相关原文片段，"
            f"请严格依据这些原文作答：\n\n{context_block}\n\n---"
        )

        if req.mode == "error_analysis":
            kp_str    = "、".join(req.knowledge_points) if req.knowledge_points else "（未标注）"
            correct   = req.correct_answer or "（正确答案）"
            student   = req.student_answer or "（学生答案）"
            error_str = req.error_type or "（未标注）"

            user_prompt = f"""{rag_header}

【学生错题】
题目：{req.question_text}
学生答案：{student}
正确答案：{correct}
涉及知识点：{kp_str}
错误类型：{error_str}
批改点评：{req.brief_comment or "（无）"}

请结合上方教材内容，按以下结构讲解：

## 📖 教材依据
（引用教材片段中与本题最相关的内容，说明正确答案「{correct}」的教材来源）

## ✅ 为什么答案是「{correct}」
（基于教材内容解释，不引入教材外知识）

## ❌「{student}」错在哪里
（结合教材范围，说明该答案为何不正确）

## 💡 解题技巧
（同类题的快速判断方法，结合教材知识点）"""

        else:  # question_explain
            kp_str  = "、".join(req.knowledge_points) if req.knowledge_points else ""
            kp_line = f"\n知识点提示：{kp_str}" if kp_str else ""

            user_prompt = f"""{rag_header}

【题目】
{req.question_text}{kp_line}

请结合上方教材内容，按以下结构讲解：

## 📖 教材知识点
（引用教材相关内容，建立解题的理论基础）

## ✅ 解题思路 & 过程
（基于教材知识点，完整解析此题）

## 💡 记忆技巧 & 拓展
（帮助记忆相关教材知识，举一反三）"""

        return system_prompt, user_prompt

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _build_retrieval_query(self, req: ExplainRequest) -> str:
        """
        构建精准的教材检索 query：
        知识点 + 答案关键词（英语场景）+ 题目摘要
        """
        parts: list[str] = []
        if req.knowledge_points:
            parts.extend(req.knowledge_points)
        if req.correct_answer and req.student_answer:
            parts.append(f"{req.correct_answer} {req.student_answer}")
        elif req.correct_answer:
            parts.append(req.correct_answer)
        parts.append(req.question_text[:60])
        return " ".join(parts)

    def _build_key_concepts(self, req: ExplainRequest) -> list[str]:
        """合并知识点 + 错误类型为前端标签，去重。"""
        seen: set[str] = set()
        result: list[str] = []
        for item in [*req.knowledge_points, req.error_type]:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    # ── 带图片的错题解析（视觉 LLM 直接看原卷）──────────────────────────────

    async def explain_with_image(
        self,
        request: ExplainRequest,
        image_bytes: bytes,
        question_number: str = "",
        content_type: str = "image/jpeg",
    ) -> ExplainResponse:
        """
        将原始作业图片 + 错题上下文一起传给视觉 LLM，
        让 LLM 直接"看图"针对学生的具体错误讲解。
        """
        subject_cn = _SUBJECT_CN.get(request.subject.lower(), request.subject)
        q_ref      = f"第 {question_number} 题" if question_number else "该错题"
        kp_str     = "、".join(request.knowledge_points) if request.knowledge_points else "（未标注）"
        error_str  = request.error_type or "（未标注）"

        system_prompt = (
            f"你是一位初中二年级{subject_cn}老师，"
            "用简洁、亲切的语言帮助 13-14 岁学生理解自己的错误。"
            "回答使用 Markdown 格式，层次清晰。"
        )

        user_prompt = (
            f"请仔细查看图片中的{subject_cn}作业，重点关注【{q_ref}】。\n\n"
            f"【批改信息】\n"
            f"学生答案：{request.student_answer or '（未作答）'}\n"
            f"正确答案：{request.correct_answer or '（见题目）'}\n"
            f"错误类型：{error_str}\n"
            f"涉及知识点：{kp_str}\n"
            f"批改点评：{request.brief_comment or '（无）'}\n\n"
            f"请**针对该学生的具体错误**，按以下结构讲解：\n\n"
            f"## 🔍 错误分析\n"
            f"（结合图片中{q_ref}的内容，具体指出学生在哪一步出错、为什么错）\n\n"
            f"## ✅ 正确解题过程\n"
            f"（完整推导步骤；数学/物理/化学要写出关键公式，不要跳步）\n\n"
            f"## 💡 避免同类错误的技巧\n"
            f"（总结该错误的规律，帮助学生下次不再犯同类错误）"
        )

        b64       = base64.b64encode(image_bytes).decode()
        data_url  = f"data:{content_type};base64,{b64}"
        messages  = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text",      "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]},
        ]

        explanation = await self.call_llm(
            user_prompt="",
            system_prompt="",
            messages=messages,
            stage="vision_error_analysis",
        )

        return ExplainResponse(
            question_text=request.question_text,
            subject=request.subject,
            mode="error_analysis",
            explanation=explanation,
            key_concepts=self._build_key_concepts(request),
            used_rag=False,
            textbook_snippets=[],
        )

    # ── 公共辅助（供 router 在多轮模式下单独调用）───────────────────────────

    async def retrieve_snippets(
        self, subject: str, query: str, top_k: int = 4
    ) -> list[str]:
        """
        独立检索教材片段，供多轮上下文（call_llm(messages=...)）注入使用。
        理科科目直接返回空列表（无教材 RAG）。
        """
        if subject.lower() in _LLM_ONLY_SUBJECTS:
            return []
        kb_name = await self._kb_manager.ensure_indexed(subject)
        if not kb_name:
            return []
        return await self._kb_manager.search(kb_name, query[:120], top_k=top_k)

    def get_textbook_desc(self, subject: str) -> str:
        """返回该科目的具体教材描述（如"人教版八年级下册历史教材"），供 prompt 注明教材版本。"""
        return self._kb_manager.get_textbook_desc(subject)
