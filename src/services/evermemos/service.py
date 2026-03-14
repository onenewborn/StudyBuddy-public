"""
EverMemOSService — 业务逻辑层
==============================

将 StudyBuddy 数据模型翻译成 EverMemOS 消息，负责写入记忆。

当前实现（第一步：只写入）：
  - log_performance()     作业/考试成绩 → EventLog + Foresight
  - log_session_summary() 会话摘要      → Episodic
  - log_preference()      学习偏好      → Profile

后续（第二步）可增加 search_context() 用于读取记忆并注入 prompt。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from src.agents.memory.profile_schema import PerformanceRecord
from src.logging import get_logger
from src.services.evermemos.client import EverMemOSClient

logger = get_logger("EverMemOSService")

_USER_ID  = os.getenv("EVERMEMOS_USER_ID", "xiaomeng_studybuddy")

_SUBJECT_CN = {
    "math": "数学", "physics": "物理", "chemistry": "化学",
    "english": "英语", "chinese": "语文", "history": "历史", "biology": "生物",
    "politics": "政治", "geography": "地理",
}
_RECORD_TYPE_CN = {"homework": "作业", "exam": "考试"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EverMemOSService:
    """
    单例服务，调用 EverMemOS Cloud API 写入/读取长期记忆。

    所有写入操作都是 fire-and-forget，失败只记 warning，不抛异常。
    """

    def __init__(self, user_id: str = "") -> None:
        self._user_id = user_id or _USER_ID
        self._client  = EverMemOSClient()
        # group_id：同一学生的所有记忆归为一组
        self._group_id = f"{self._user_id}_study"

    # ─────────────────────────────────────────────────────────────
    # 写入：作业 / 考试成绩
    # ─────────────────────────────────────────────────────────────

    async def log_performance(self, record: PerformanceRecord) -> None:
        """
        将一次作业/考试成绩写入 EverMemOS，触发：
          1. EventLog — 成绩事实（题数/正确率/错误类型）
          2. Foresight — 薄弱知识点复习提醒（若存在）
        """
        subject_cn     = _SUBJECT_CN.get(record.subject, record.subject)
        record_type_cn = _RECORD_TYPE_CN.get(record.record_type, record.record_type)
        exam_label     = f"「{record.exam_name}」" if record.exam_name.strip() else ""

        # ── 1. EventLog：成绩原子事实 ────────────────────────────
        score_part = ""
        if record.earned_score is not None and record.total_score:
            score_part = f"得分 {record.earned_score}/{record.total_score}，"

        error_parts: list[str] = []
        # 这里没有 error_counts，只能从 record 字段还原摘要
        if record.wrong_count > 0:
            error_parts.append(f"错误 {record.wrong_count} 题")
        if record.partial_count > 0:
            error_parts.append(f"部分正确 {record.partial_count} 题")
        if record.blank_count > 0:
            error_parts.append(f"未作答 {record.blank_count} 题")
        error_summary = "，".join(error_parts) + "。" if error_parts else "全部完成。"

        perf_content = (
            f"【系统批改结果】"
            f"学生完成了{subject_cn}{record_type_cn}{exam_label}，"
            f"共 {record.total_questions} 题，"
            f"正确 {record.correct_count} 题，"
            f"{error_summary}"
            f"{score_part}"
            f"正确率 {record.accuracy_rate:.0%}。"
            f"记录时间：{record.recorded_at}。"
        )

        result = await self._client.add_message(
            sender=self._user_id,
            content=perf_content,
            group_id=self._group_id,
            group_name="StudyBuddy 学习记录",
            sender_name="批改系统",
            flush=True,
        )
        logger.info(
            f"[EverMemOS] EventLog 写入: subject={record.subject} "
            f"accuracy={record.accuracy_rate:.0%} "
            f"status={result.get('status')}"
        )

        # ── 2. Foresight：薄弱知识点复习提醒 ────────────────────
        if record.weak_knowledge_points:
            points_str = "、".join(record.weak_knowledge_points[:8])
            foresight_content = (
                f"【学情诊断】"
                f"系统批改检测到学生在{subject_cn}上存在薄弱知识点，"
                f"建议重点复习：{points_str}。"
                f"（来源：{record_type_cn}{exam_label}，{record.recorded_at[:10]}）"
            )
            result2 = await self._client.add_message(
                sender=self._user_id,
                content=foresight_content,
                group_id=self._group_id,
                group_name="StudyBuddy 学习记录",
                sender_name="批改系统",
                flush=True,
            )
            logger.info(
                f"[EverMemOS] Foresight 写入: {len(record.weak_knowledge_points)} 个薄弱点 "
                f"status={result2.get('status')}"
            )

    # ─────────────────────────────────────────────────────────────
    # 写入：错误类型分布
    # ─────────────────────────────────────────────────────────────

    async def log_error_patterns(
        self, subject: str, error_counts: dict[str, int]
    ) -> None:
        """
        将本次作业暴露的错误类型写入 EverMemOS EventLog。
        """
        significant = {k: v for k, v in error_counts.items() if v > 0}
        if not significant:
            return

        subject_cn = _SUBJECT_CN.get(subject, subject)
        _ERROR_CN = {
            "calculation_error": "计算失误",
            "concept_confusion": "概念混淆",
            "reading_mistake":   "审题失误",
            "formula_wrong":     "公式错误",
            "sign_error":        "正负号错误",
            "unit_error":        "单位错误",
            "incomplete":        "步骤不完整",
            "logic_error":       "逻辑错误",
            "spelling_grammar":  "拼写/语法",
            "other":             "其他",
        }
        parts = [
            f"{_ERROR_CN.get(k, k)} {v} 次"
            for k, v in significant.items()
        ]
        content = (
            f"【错误类型统计】"
            f"系统批改检测到学生本次{subject_cn}作业的错误类型分布：{'，'.join(parts)}。"
        )
        result = await self._client.add_message(
            sender=self._user_id,
            content=content,
            group_id=self._group_id,
            group_name="StudyBuddy 学习记录",
            sender_name="批改系统",
            flush=True,
        )
        logger.info(
            f"[EverMemOS] 错误类型写入: {parts} "
            f"status={result.get('status')}"
        )

    # ─────────────────────────────────────────────────────────────
    # 写入：会话摘要（Episodic）
    # ─────────────────────────────────────────────────────────────

    async def log_session_summary(
        self,
        subject: str,
        session_id: str,
        summary: str,
        turn_count: int = 0,
    ) -> None:
        """
        将 LLM 压缩后的会话摘要写入 EverMemOS Episodic 记忆。
        在 explain.py 会话压缩时调用。

        摘要格式应为双段结构（由 ChatCompressor 保证）：
          【学生提问】... ← 以学生 sender 写入，反映真实提问行为
          【学情分析】... ← 以学情系统 sender 写入，标明是 AI 分析结论
        若格式不符（旧摘要兼容）则整体以"学情系统"写入，避免错误归因。
        """
        if not summary.strip():
            return

        subject_cn = _SUBJECT_CN.get(subject, subject)
        session_tag = f"（session={session_id[:8]}，共 {turn_count} 轮）"

        STUDENT_TAG  = "【学生提问】"
        ANALYSIS_TAG = "【学情分析】"

        if STUDENT_TAG in summary and ANALYSIS_TAG in summary:
            # ── 结构化摘要：拆分后分别归因写入 ──────────────────────
            idx_s = summary.index(STUDENT_TAG)
            idx_a = summary.index(ANALYSIS_TAG)

            if idx_s < idx_a:
                student_part  = summary[idx_s + len(STUDENT_TAG):idx_a].strip()
                analysis_part = summary[idx_a + len(ANALYSIS_TAG):].strip()
            else:
                analysis_part = summary[idx_a + len(ANALYSIS_TAG):idx_s].strip()
                student_part  = summary[idx_s + len(STUDENT_TAG):].strip()

            # 学生提问部分 — 真实反映学生行为，用"学生" sender
            if student_part:
                r1 = await self._client.add_message(
                    sender=self._user_id,
                    content=f"【学生提问】{subject_cn}学习会话{session_tag}：{student_part}",
                    group_id=self._group_id,
                    group_name="StudyBuddy 学习记录",
                    sender_name="学生",
                    flush=False,
                )
                logger.info(
                    f"[EverMemOS] 学生提问摘要写入: session={session_id[:8]} "
                    f"status={r1.get('status')}"
                )

            # 学情分析部分 — AI 系统结论，用"学情系统" sender
            if analysis_part:
                r2 = await self._client.add_message(
                    sender=self._user_id,
                    content=f"【学情分析】{subject_cn}学习会话{session_tag}系统分析：{analysis_part}",
                    group_id=self._group_id,
                    group_name="StudyBuddy 学习记录",
                    sender_name="学情系统",
                    flush=True,
                )
                logger.info(
                    f"[EverMemOS] 学情分析摘要写入: session={session_id[:8]} "
                    f"status={r2.get('status')}"
                )
        else:
            # ── 旧格式兼容：整体归因给"学情系统"，避免错误推断学生偏好 ──
            content = (
                f"【学情分析】{subject_cn}学习会话{session_tag}摘要：{summary}"
            )
            result = await self._client.add_message(
                sender=self._user_id,
                content=content,
                group_id=self._group_id,
                group_name="StudyBuddy 学习记录",
                sender_name="学情系统",
                flush=True,
            )
            logger.info(
                f"[EverMemOS] Episodic 写入（旧格式）: session={session_id[:8]} "
                f"turns={turn_count} status={result.get('status')}"
            )

    # ─────────────────────────────────────────────────────────────
    # 读取：检索相关记忆（第二步用，目前预留）
    # ─────────────────────────────────────────────────────────────

    async def log_wrong_question(
        self,
        subject: str,
        number: str,
        question_text: str,
        student_answer: str,
        correct_answer: str,
        error_type: str = "",
        brief_comment: str = "",
    ) -> None:
        """
        将一道错题的详情写入 EverMemOS，供后续学情分析和针对性讲解使用。
        """
        subject_cn = _SUBJECT_CN.get(subject, subject)
        _ERROR_CN = {
            "calculation_error": "计算失误", "concept_confusion": "概念混淆",
            "reading_mistake": "审题失误", "formula_wrong": "公式错误",
            "sign_error": "正负号错误", "unit_error": "单位错误",
            "incomplete": "步骤不完整", "logic_error": "逻辑错误",
            "spelling_grammar": "拼写/语法", "other": "其他",
        }
        error_str = _ERROR_CN.get(error_type, error_type) if error_type else "未分类"
        comment_str = f"，点评：{brief_comment}" if brief_comment else ""

        content = (
            f"【错题记录】"
            f"系统批改检测到学生做错了一道{subject_cn}题（题{number}）。\n"
            f"题目：{question_text[:300]}\n"
            f"学生作答：{student_answer[:150] or '（空白）'}\n"
            f"正确答案：{correct_answer[:150]}\n"
            f"错误类型：{error_str}{comment_str}"
        )
        result = await self._client.add_message(
            sender=self._user_id,
            content=content,
            group_id=self._group_id,
            group_name="StudyBuddy 学习记录",
            sender_name="批改系统",
            flush=False,   # 多条错题批量写入，不每条都触发提取
        )
        logger.debug(
            f"[EverMemOS] 错题写入: 题{number} {subject} {error_str} "
            f"status={result.get('status')}"
        )

    # ─────────────────────────────────────────────────────────────
    # 写入：用户个人档案（长期记忆 Profile）
    # ─────────────────────────────────────────────────────────────

    async def log_user_profile(self, pref: dict) -> None:
        """
        将用户个人信息写入 EverMemOS 作为长期 Profile 记忆。
        每次用户修改偏好设置时调用，EverMemOS 会更新其用户画像。

        Args:
            pref: UserPreferences.model_dump() 字典
        """
        parts: list[str] = []

        if pref.get("nickname"):
            parts.append(f"昵称：{pref['nickname']}")
        if pref.get("grade"):
            parts.append(f"年级：{pref['grade']}")
        if pref.get("gender"):
            gender_cn = {"male": "男生", "female": "女生"}.get(pref["gender"], "")
            if gender_cn:
                parts.append(f"性别：{gender_cn}")
        if pref.get("mbti"):
            parts.append(f"MBTI性格：{pref['mbti']}")

        length_map = {"brief": "简短精炼", "standard": "适中", "detailed": "详细深入"}
        style_map  = {"vivid": "生动形象", "objective": "简洁客观"}
        length_cn  = length_map.get(pref.get("explanation_length", ""), "适中")
        style_cn   = style_map.get(pref.get("explanation_style", ""), "生动形象")
        parts.append(f"讲解风格：{style_cn}，详细程度：{length_cn}")

        if not parts:
            return

        content = "【学生档案】" + "；".join(parts) + "。"

        result = await self._client.add_message(
            sender=self._user_id,
            content=content,
            group_id=self._group_id,
            group_name="StudyBuddy 学习记录",
            sender_name="学生",
            flush=True,   # 档案是独立事件，立即触发记忆提取
        )
        logger.info(
            f"[EverMemOS] 用户档案写入: {content[:80]}… "
            f"status={result.get('status')}"
        )

    async def search_context(
        self,
        subject: str,
        query: Optional[str] = None,
        top_k: int = 5,
    ) -> list[str]:
        """
        检索与当前学习情境相关的记忆，返回文本列表（用于注入 prompt）。
        """
        effective_query = query or f"{_SUBJECT_CN.get(subject, subject)} 学习情况 薄弱点"
        resp = await self._client.search(
            query=effective_query,
            user_id=self._user_id,
            memory_types=["event_log", "foresight", "episodic_memory"],
            top_k=top_k,
        )

        # 兼容两种响应格式：
        # 1. {"memories": [{"group_id":…, "memories":[…]}]}  ← 嵌套分组格式
        # 2. {"result": {"memories": […]}}                   ← 旧格式
        # 3. {"status": "error", …}                          ← 报错
        if resp.get("status") == "error" or not resp:
            return []

        texts: list[str] = []
        raw = resp.get("memories") or resp.get("result", {}).get("memories", [])

        for item in raw:
            if isinstance(item, dict) and "memories" in item:
                # 嵌套分组：{"group_id":…, "memories":[…]}
                for m in item["memories"]:
                    text = m.get("summary") or m.get("episode") or m.get("content", "")
                    if text:
                        texts.append(text.strip())
            else:
                # 平铺格式
                text = item.get("summary") or item.get("episode") or item.get("content", "")
                if text:
                    texts.append(text.strip())

        return texts[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 单例
# ─────────────────────────────────────────────────────────────────────────────

_instance: Optional[EverMemOSService] = None


def get_evermemos_service() -> EverMemOSService:
    global _instance
    if _instance is None:
        _instance = EverMemOSService()
    return _instance
