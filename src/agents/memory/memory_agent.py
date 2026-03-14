"""
ChatCompressor — 基于 LLM 的聊天会话压缩智能体
=================================================

继承 BaseAgent 以复用 call_llm() 接口。
达到 COMPRESSION_THRESHOLD 轮问答后自动触发，将会话历史压缩为摘要。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.agents.base_agent import BaseAgent
from src.agents.memory.profile_schema import ChatSession
from src.logging import get_logger

logger = get_logger("ChatCompressor")


class ChatCompressor(BaseAgent):
    """将聊天会话历史 LLM 压缩为简洁摘要的智能体。"""

    def __init__(self) -> None:
        # 复用 explain_agent 的 LLM 配置（model / temperature / max_tokens）
        super().__init__(module_name="explain", agent_name="explain_agent")

    async def process(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        raise NotImplementedError

    async def compress(self, session: ChatSession) -> str:
        """
        将 session 的历史记录压缩为一段中文摘要（200 字以内）。
        优先使用 turn_records（文本更完整）；若无则回退到 messages。
        失败时返回空字符串（不抛异常）。
        """
        # 优先使用 turn_records（含完整用户提问和 AI 回复）
        if session.turn_records:
            lines: list[str] = []
            for turn in session.turn_records:
                lines.append(f"学生：{turn.user_text[:300]}")
                lines.append(f"AI老师：{turn.assistant_text[:300]}")
            conversation = "\n".join(lines)
            turns_count = len(session.turn_records)
        elif session.messages:
            lines = []
            for msg in session.messages:
                role_cn = "学生" if msg.role == "user" else "AI老师"
                lines.append(f"{role_cn}：{msg.content[:200]}")
            conversation = "\n".join(lines)
            turns_count = session.turn_count
        else:
            return ""

        system_prompt = (
            "你是一名学习分析助手，专门处理初中生的学习对话记录。\n"
            "请将以下师生对话压缩为两段结构化摘要，格式严格如下（不要增减标签）：\n\n"
            "【学生提问】（100字以内）\n"
            "只记录学生自己说了什么、问了什么、在哪里卡住或表示不理解。"
            "禁止混入 AI 的解释内容，禁止推断学生偏好。\n\n"
            "【学情分析】（100字以内）\n"
            "基于对话，分析学生暴露的薄弱知识点和错误模式，这是系统分析结论，非学生原话。\n\n"
            "用中文输出，直接给出两段正文，禁止复述原对话。"
        )
        user_prompt = (
            f"以下是本次学习会话记录（共 {turns_count} 轮问答）：\n\n"
            f"{conversation}\n\n"
            "请按格式生成摘要："
        )

        try:
            summary = await self.call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                stage="compress_session",
            )
            summary = summary.strip()
            logger.info(
                f"[ChatCompressor] session={session.session_id} "
                f"compressed to {len(summary)} chars"
            )
            return summary
        except Exception as exc:
            logger.warning(f"[ChatCompressor] compression failed: {exc}")
            return ""
