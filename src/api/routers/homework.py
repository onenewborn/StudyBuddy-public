"""
Homework Grading API Router
============================

Endpoints
---------
POST /api/v1/homework/grade    — Upload image(s) + subject, run full 4-step pipeline
GET  /api/v1/homework/health   — Service health check

Pipeline
--------
图片上传
  ↓ OCRAgent        — 识别题目 + 学生作答（多图并行）→ list[ExtractedQuestion]
  ↓ GradeAgent      — 逐题批改                      → list[GradedQuestion]
  ↓ HomeworkResult  — 汇总返回（以下后台执行）
  ↓ KnowPointAgent  — 知识点 + 难度标注（后台）     → list[GradedQuestion]（已注解）
  ↓ ExamTagAgent    — 试卷特征 + 薄弱知识点（后台）  → (exam_tags, weak_points)
"""

from __future__ import annotations

import asyncio
import base64
import sys
from datetime import datetime
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# ── Project root in path ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.homework import (
    ExamTagAgent,
    GradeAgent,
    HomeworkResult,
    KnowPointAgent,
    OCRAgent,
)
from src.agents.homework.models import GradedQuestion, GradeResult, QuestionType
from src.agents.homework.wrong_book_service import WrongBookService
from src.agents.memory import MemoryService, PerformanceRecord
from src.config.model_registry import agent_kwargs
from src.logging import get_logger
from src.services.evermemos import get_evermemos_service
from src.services.question_bank import QuestionBankService

# ── Logger ────────────────────────────────────────────────────────────────────
logger = get_logger("Homework")

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter()

# ── Memory service singleton ──────────────────────────────────────────────────
_mem_service: "MemoryService | None" = None

_ERR_LABEL_CN: dict[str, str] = {
    "calculation_error": "计算失误",
    "concept_confusion": "概念混淆",
    "reading_mistake":   "审题失误",
    "formula_wrong":     "公式错误",
    "sign_error":        "符号错误",
    "unit_error":        "单位错误",
    "incomplete":        "解答不完整",
    "logic_error":       "逻辑错误",
    "spelling_grammar":  "拼写/语法错误",
    "other":             "其他",
}


async def _build_grade_memory_hint(subject: str) -> tuple[str, list[str]]:
    """
    从记忆服务构造注入 GradeAgent 的 prompt 片段，同时返回薄弱知识点列表。

    Returns:
        (memory_hint_str, known_weak_points)
        - memory_hint_str  : 直接拼接到 GradeAgent system prompt 末尾
        - known_weak_points: 传给 KnowPointAgent 用于 is_weak_area 标注
    """
    try:
        mem_svc = _get_mem_service()
        memory = await mem_svc.load_memory()

        parts: list[str] = []

        # 高频错误类型（Top 3）
        err_counts = memory.error_pattern_counts.get(subject, {})
        if err_counts:
            top_errors = sorted(err_counts, key=lambda k: err_counts[k], reverse=True)[:3]
            errs_cn = "、".join(_ERR_LABEL_CN.get(e, e) for e in top_errors)
            parts.append(f"该学生在本科目的高频错误：{errs_cn}，请结合历史模式精准识别 error_type")

        # 近期薄弱知识点（最近 6 个）
        weak_points = memory.knowledge_gaps.get(subject, [])[-6:]
        if weak_points:
            pts = "、".join(weak_points)
            parts.append(
                f"近期薄弱知识点：{pts}；"
                f"若该题涉及上述知识点，brief_comment 中可点出「注意这是你的薄弱环节」"
            )

        if not parts:
            return "", []

        hint = (
            "\n\n【学生历史档案（仅供参考，不影响判题标准）】\n"
            + "\n".join(f"• {p}" for p in parts)
        )
        return hint, weak_points

    except Exception as e:
        logger.warning(f"[Homework] 记忆档案加载失败（不影响批改）: {e}")
        return "", []

def _get_mem_service() -> MemoryService:
    global _mem_service
    if _mem_service is None:
        _mem_service = MemoryService(_project_root / "data")
    return _mem_service

# ── Question bank singleton ────────────────────────────────────────────────────
_question_bank: "QuestionBankService | None" = None

def _get_question_bank() -> QuestionBankService:
    global _question_bank
    if _question_bank is None:
        _question_bank = QuestionBankService(_project_root / "data")
    return _question_bank

# 允许的科目白名单（前端 select 对齐）
_VALID_SUBJECTS = {
    "math",      # 数学
    "physics",   # 物理
    "chemistry", # 化学
    "english",   # 英语
    "chinese",   # 语文
    "history",   # 历史
    "biology",   # 生物
    "politics",  # 政治
    "geography", # 地理
}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """作业批改模块健康检查。"""
    return {"status": "ok", "module": "homework"}


@router.get("/question-bank/stats")
async def question_bank_stats(
    subject: str = "math",
):
    """题库统计：各科目缓存的题目数量和命中次数。"""
    bank = _get_question_bank()
    if subject == "all":
        subjects = list(_VALID_SUBJECTS)
        return [bank.stats(s) for s in subjects]
    return bank.stats(subject)


@router.post("/grade", response_model=HomeworkResult)
async def grade_homework(
    files: list[UploadFile] = File(
        ...,
        description="试卷图片（JPG / PNG / WEBP），支持多张，顺序即页码顺序",
    ),
    subject: str = Form(
        default="math",
        description="科目：math | physics | chemistry | english | chinese | history | biology",
    ),
    record_type: str = Form(
        default="homework",
        description="类型：homework（作业）| exam（考试）",
    ),
    exam_name: str = Form(
        default="",
        description="考试名称（record_type=exam 时填写，如「期中考试」）",
    ),
    model_key: str = Form(
        default="",
        description="模型预设键：gemini | kimi（不传则使用系统默认）",
    ),
):
    """
    作业批改完整流程（同步，等待全部结果后返回）。

    接收一张或多张试卷图片，经过四个 AI 阶段处理后返回完整结构化结果：

    1. **OCR**        — 识别所有题目及学生作答
    2. **Grade**      — 逐题批改，给出正确答案和点评
    3. **KnowPoint**  — 标注每题知识点和难度等级
    4. **ExamTag**    — 分析整份试卷特征 + 找出薄弱知识点

    **Request (multipart/form-data)**
    - `files`   : 图片文件，至少 1 张
    - `subject` : 科目字符串（默认 `math`）

    **Response**
    - `HomeworkResult` JSON：含题目明细、批改统计、试卷标签、薄弱知识点
    """
    # ── 0. 验证科目 ─────────────────────────────────────────────────────────
    subject_key = subject.strip().lower()
    if subject_key not in _VALID_SUBJECTS:
        logger.warning(f"[Homework] 未知科目 '{subject}'，回退为 math")
        subject_key = "math"

    # ── 1. 读取文件，转 base64 ───────────────────────────────────────────────
    images: list[str] = []
    for upload in files:
        content = await upload.read()
        if not content:
            continue
        b64 = base64.b64encode(content).decode()
        images.append(b64)

    if not images:
        raise HTTPException(status_code=400, detail="请上传至少一张试卷图片")

    logger.info(f"[Homework] 收到 {len(images)} 张图片，科目={subject_key}")

    # ── 保存第一张图片到磁盘（供错题解析时传给视觉 LLM）────────────────────
    saved_image_path: Optional[str] = None
    try:
        img_dir = _project_root / "data" / "homework_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = img_dir / f"{ts}_{subject_key}.jpg"
        img_file.write_bytes(base64.b64decode(images[0]))
        saved_image_path = str(img_file)
    except Exception as img_exc:
        logger.warning(f"[Homework] 图片保存失败（不影响批改）: {img_exc}")

    # 用户未指定模型时默认走 kimi；OCR 与后续步骤统一使用同一个模型
    effective_key = model_key.strip() or "kimi"
    model_kw = agent_kwargs(effective_key) or {}
    ocr_kw = model_kw

    try:
        # ── 2. OCR ─────────────────────────────────────────────────────────
        logger.info("[Homework] 第一步：OCR 识别中（多图并行）...")
        ocr_agent = OCRAgent(parse_model_key=effective_key, **ocr_kw)
        questions = await ocr_agent.process(images=images, subject=subject_key)

        if not questions:
            raise HTTPException(
                status_code=422,
                detail="未能从图片中识别出任何题目，请检查图片质量或方向",
            )
        logger.info(f"[Homework] OCR 完成，共识别 {len(questions)} 道题")

        # ── 2.5 加载记忆档案（一次读取，分别注入 Grade 和 KnowPoint）──────
        memory_hint, known_weak_points = await _build_grade_memory_hint(subject_key)

        # ── 3. Grade（题库缓存优先，命中则跳过 LLM）──────────────────────
        logger.info("[Homework] 第二步：查题库 + 批改中...")
        bank = _get_question_bank()

        # 对每道题查题库，客观题命中则直接构造 GradedQuestion
        cached_map: dict[int, GradedQuestion] = {}
        miss_questions = []
        miss_indices: list[int] = []

        for i, q in enumerate(questions):
            qtype = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
            hit = await bank.lookup(q.question_text, q.student_answer, qtype, subject_key)
            if hit:
                grade_val = hit["grade"]
                try:
                    grade_enum = GradeResult(grade_val)
                except ValueError:
                    grade_enum = GradeResult.SKIP

                # 推算 earned_score
                earned = None
                if q.score_value is not None:
                    if grade_enum == GradeResult.CORRECT:
                        earned = q.score_value
                    elif grade_enum in (GradeResult.WRONG, GradeResult.BLANK):
                        earned = 0.0

                cached_map[i] = GradedQuestion(
                    number=q.number,
                    question_text=q.question_text,
                    student_answer=q.student_answer,
                    question_type=q.question_type,
                    score_value=q.score_value,
                    page_index=q.page_index,
                    correct_answer=hit["correct_answer"],
                    grade=grade_enum,
                    earned_score=earned,
                    error_type=None,
                    brief_comment=hit["brief_comment"],
                    knowledge_points=[],
                    difficulty=None,
                )
            else:
                miss_questions.append(q)
                miss_indices.append(i)

        # 只对题库未命中的题调用 LLM
        newly_graded: list[GradedQuestion] = []
        if miss_questions:
            grade_agent = GradeAgent(**(model_kw or {}))
            newly_graded = await grade_agent.process(
                miss_questions, subject=subject_key, memory_hint=memory_hint
            )

        # 合并结果，保持原始顺序
        graded_slots: list[GradedQuestion | None] = [None] * len(questions)
        for idx, gq in cached_map.items():
            graded_slots[idx] = gq
        for list_i, orig_idx in enumerate(miss_indices):
            if list_i < len(newly_graded):
                graded_slots[orig_idx] = newly_graded[list_i]
        graded = [g for g in graded_slots if g is not None]

        cache_hits = len(cached_map)
        if cache_hits:
            logger.info(
                f"[Homework] 题库命中 {cache_hits}/{len(questions)} 题，"
                f"LLM 仅批改 {len(miss_questions)} 题"
            )
        logger.info("[Homework] 批改完成")

        # ── 4. 汇总核心结果（KnowPoint / ExamTag 后台执行）─────────────
        result = HomeworkResult(
            subject=subject_key,
            image_count=len(images),
            exam_tags=[],              # ExamTag 异步后台填充，响应时为空
            weak_knowledge_points=[],  # 同上
            questions=graded,          # KnowPoint 后台注解，响应时无知识点
        )
        result.compute_stats()

        logger.info(
            f"[Homework] 核心批改完成（ExamTag 后台执行）| "
            f"总题数={result.total_questions} "
            f"✓{result.correct_count} "
            f"✗{result.wrong_count} "
            f"△{result.partial_count} "
            f"○{result.blank_count}"
        )

        # ── 6. 自动存档错题（同步，重要）────────────────────────────────
        try:
            wb_svc = WrongBookService(_project_root / "data")
            saved = await wb_svc.save_from_result(result, source_image_path=saved_image_path)
            if saved:
                logger.info(f"[Homework] 错题本已存档 {saved} 条新错题")
        except Exception as wb_exc:
            logger.warning(f"[Homework] 错题本存档失败（不影响结果）: {wb_exc}")

        # ── 7. ExamTag + 记忆系统 → 后台异步，不阻塞响应 ────────────────
        record_type_key = record_type.strip().lower()
        if record_type_key not in {"homework", "exam"}:
            record_type_key = "homework"

        asyncio.create_task(_background_exam_and_memory(
            graded=graded,
            newly_graded=newly_graded,   # 仅 LLM 批改的题写入题库
            subject_key=subject_key,
            record_type_key=record_type_key,
            exam_name=exam_name.strip(),
            result=result,
            model_kw=model_kw or {},
            known_weak_points=known_weak_points,
        ))

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[Homework] 批改过程出错: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批改失败：{exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# 后台任务：ExamTag + 记忆系统（不阻塞批改响应）
# ─────────────────────────────────────────────────────────────────────────────

async def _save_homework_history(result: "HomeworkResult") -> None:
    """将完整批改结果持久化到 data/history/homework/{id}.json。"""
    try:
        import json as _json
        from datetime import datetime as _dt
        hist_dir = _project_root / "data" / "history" / "homework"
        hist_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        record_id = f"{ts}_{result.subject}"
        path = hist_dir / f"{record_id}.json"
        path.write_text(
            _json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[Homework/bg] 批改历史已保存: {record_id}.json")
    except Exception as e:
        logger.warning(f"[Homework/bg] 批改历史保存失败（忽略）: {e}")


async def _background_exam_and_memory(
    graded: list,
    newly_graded: list,   # 仅 LLM 本次批改的（不含题库命中的），写入题库
    subject_key: str,
    record_type_key: str,
    exam_name: str,
    result: "HomeworkResult",
    model_kw: dict,
    known_weak_points: list[str] | None = None,
) -> None:
    """
    异步后台：题库写入 → KnowPoint 注解 → ExamTag → 更新记忆系统 → 保存历史记录。
    批改响应已返回，此任务在后台安静执行，失败不影响用户。
    """
    exam_tags: list[str] = []
    weak_points: list[str] = []

    # ── 0. 题库写入（只保存本次 LLM 新批改的题）─────────────────────
    try:
        bank = _get_question_bank()
        saved = await bank.save_batch(newly_graded, subject_key)
        if saved:
            logger.info(f"[Homework/bg] 题库写入 {saved} 道新题")
    except Exception as e:
        logger.warning(f"[Homework/bg] 题库写入失败（忽略）: {e}")

    # ── KnowPoint：知识点 + 难度（后台注解，直接 mutate graded 对象）────
    annotated = graded
    try:
        kp_agent = KnowPointAgent(**model_kw)
        annotated = await kp_agent.process(
            graded, subject=subject_key, known_weak_points=known_weak_points or []
        )
        # 回填到 result.questions（result 对象仍被 _save_homework_history 引用）
        result.questions = annotated
        logger.info("[Homework/bg] KnowPoint 注解完成")
    except Exception as e:
        logger.warning(f"[Homework/bg] KnowPoint 失败（忽略）: {e}")

    # ExamTag
    try:
        tag_agent = ExamTagAgent(**model_kw)
        exam_tags, weak_points = await tag_agent.process(annotated, subject=subject_key)
        logger.info(
            f"[Homework/bg] ExamTag 完成: tags={exam_tags} weak={weak_points}"
        )
    except Exception as e:
        logger.warning(f"[Homework/bg] ExamTag 失败（忽略）: {e}")

    # 记忆系统
    try:
        accuracy = (
            result.correct_count / result.total_questions
            if result.total_questions > 0 else 0.0
        )
        perf = PerformanceRecord(
            record_type=record_type_key,
            subject=subject_key,
            exam_name=exam_name,
            total_questions=result.total_questions,
            correct_count=result.correct_count,
            wrong_count=result.wrong_count,
            partial_count=result.partial_count,
            blank_count=result.blank_count,
            earned_score=result.earned_score,
            total_score=result.total_score,
            accuracy_rate=round(accuracy, 4),
            weak_knowledge_points=weak_points,
            exam_tags=exam_tags,
        )
        mem_svc = _get_mem_service()
        await mem_svc.log_performance(perf)

        error_counts: dict[str, int] = {}
        for q in result.questions:
            if q.error_type:
                et = q.error_type.value if hasattr(q.error_type, "value") else str(q.error_type)
                error_counts[et] = error_counts.get(et, 0) + 1
        if error_counts:
            await mem_svc.update_error_patterns(subject_key, error_counts)

        logger.info(f"[Homework/bg] 本地记忆系统已更新 accuracy={accuracy:.0%}")
    except Exception as e:
        logger.warning(f"[Homework/bg] 本地记忆系统更新失败（忽略）: {e}")

    # ── EverMemOS：长期记忆云端写入 ──────────────────────────────
    try:
        accuracy = (
            result.correct_count / result.total_questions
            if result.total_questions > 0 else 0.0
        )
        perf_for_evermemos = PerformanceRecord(
            record_type=record_type_key,
            subject=subject_key,
            exam_name=exam_name,
            total_questions=result.total_questions,
            correct_count=result.correct_count,
            wrong_count=result.wrong_count,
            partial_count=result.partial_count,
            blank_count=result.blank_count,
            earned_score=result.earned_score,
            total_score=result.total_score,
            accuracy_rate=round(accuracy, 4),
            weak_knowledge_points=weak_points,
            exam_tags=exam_tags,
        )
        evermemos = get_evermemos_service()
        await evermemos.log_performance(perf_for_evermemos)

        error_counts_for_em: dict[str, int] = {}
        for q in result.questions:
            if q.error_type:
                et = q.error_type.value if hasattr(q.error_type, "value") else str(q.error_type)
                error_counts_for_em[et] = error_counts_for_em.get(et, 0) + 1
        if error_counts_for_em:
            await evermemos.log_error_patterns(subject_key, error_counts_for_em)

        # 每道错题详情写入（帮助 EverMemOS 建立细粒度错题记忆）
        from src.agents.homework.models import GradeResult
        wrong_qs = [
            q for q in result.questions
            if q.grade in (GradeResult.WRONG, GradeResult.PARTIAL)
        ]
        for q in wrong_qs[:10]:   # 最多写 10 道，避免超量
            et = q.error_type.value if q.error_type and hasattr(q.error_type, "value") else ""
            await evermemos.log_wrong_question(
                subject=subject_key,
                number=q.number,
                question_text=q.question_text,
                student_answer=q.student_answer or "",
                correct_answer=q.correct_answer or "",
                error_type=et,
                brief_comment=q.brief_comment or "",
            )

        logger.info(
            f"[Homework/bg] EverMemOS 写入完成 "
            f"（成绩+{len(wrong_qs)}道错题详情）"
        )
    except Exception as e:
        logger.warning(f"[Homework/bg] EverMemOS 写入失败（忽略）: {e}")

    # ── 历史记录持久化 ────────────────────────────────────────────
    # 在 exam_tags / weak_points 回填后再保存，确保记录完整
    result.exam_tags = exam_tags
    result.weak_knowledge_points = weak_points
    await _save_homework_history(result)


# ─────────────────────────────────────────────────────────────────────────────
# Regrade (appeal) endpoint
# ─────────────────────────────────────────────────────────────────────────────

class RegradeRequest(BaseModel):
    question_text: str
    student_answer: str
    subject: str
    question_type: str = "unknown"
    correct_answer: str = ""
    model_key: Optional[str] = None


@router.post("/regrade-question")
async def regrade_question(body: RegradeRequest):
    """
    **申诉重新评分**：对单道题让 AI 独立重新审阅，返回新的批改结论。

    适用场景：学生认为 AI 判错，希望 AI 重新审题确认是否确实有误。

    返回：
    - `grade`: 新判断结果（correct / wrong / partial / blank / skip）
    - `correct_answer`: 标准答案
    - `brief_comment`: 一句话点评
    - `error_type`: 错误类型（仅 wrong/partial 时有值）
    - `overturned`: 是否推翻了原判决（新结果为 correct 则为 true）
    """
    if body.subject.strip().lower() not in _VALID_SUBJECTS:
        raise HTTPException(status_code=400, detail=f"未知科目: {body.subject}")

    try:
        model_kw = agent_kwargs(body.model_key)
        agent = GradeAgent(**(model_kw or {}))
        result = await agent.regrade_single(
            question_text=body.question_text,
            student_answer=body.student_answer,
            subject=body.subject.strip().lower(),
            question_type=body.question_type,
            correct_answer=body.correct_answer,
        )
        result["overturned"] = result.get("grade") == "correct"
        return result
    except Exception as exc:
        logger.error(f"[Homework] 申诉评分失败: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重新评分失败：{exc!s}")
