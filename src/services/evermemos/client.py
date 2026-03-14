"""
EverMemOS API Client
====================

封装对 https://api.evermind.ai 的底层 HTTP 调用。

API 参考：
  POST /api/v0/memories        — 写入一条消息（异步提取）
  GET  /api/v0/memories/search — 语义搜索记忆
  GET  /api/v0/memories        — 列举记忆
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from src.logging import get_logger

logger = get_logger("EverMemOSClient")

_BASE_URL = os.getenv("EVERMEMOS_BASE_URL", "https://api.evermind.ai")
_API_KEY  = os.getenv("EVERMEMOS_API_KEY", "")
_TIMEOUT  = 20  # 秒


class EverMemOSClient:
    """
    EverMemOS REST API 的轻量异步客户端。

    所有方法失败时只记录 warning，不抛异常，
    保证调用方（后台任务）不会因此崩溃。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
    ) -> None:
        self._api_key  = api_key  or _API_KEY
        self._base_url = (base_url or _BASE_URL).rstrip("/")
        self._headers  = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    # ─────────────────────────────────────────
    # Write — 写入一条消息
    # ─────────────────────────────────────────

    async def add_message(
        self,
        sender: str,
        content: str,
        *,
        role: str = "user",
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        sender_name: Optional[str] = None,
        flush: bool = True,
        message_id: Optional[str] = None,
        create_time: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        向 EverMemOS 写入一条消息，触发异步记忆提取。

        Args:
            sender:      消息发送者 ID（学生唯一标识）
            content:     消息正文
            role:        "user" | "assistant"
            group_id:    组织 / 群组 ID（留空则 API 自动生成）
            flush:       True = 立即触发记忆提取边界（本条消息为一个完整事件）

        Returns:
            {"status": "queued", "request_id": "..."}
        """
        payload: dict[str, Any] = {
            "message_id":  message_id  or str(uuid.uuid4()),
            "create_time": create_time or datetime.now(timezone.utc).isoformat(),
            "sender":      sender,
            "content":     content,
            "role":        role,
            "flush":       flush,
        }
        if group_id:
            payload["group_id"] = group_id
        if group_name:
            payload["group_name"] = group_name
        if sender_name:
            payload["sender_name"] = sender_name

        return await self._post("/api/v0/memories", payload)

    # ─────────────────────────────────────────
    # Search — 语义检索记忆
    # ─────────────────────────────────────────

    async def search(
        self,
        query: str,
        user_id: str,
        *,
        memory_types: Optional[list[str]] = None,
        top_k: int = 5,
        retrieve_method: str = "hybrid",
        group_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        按语义查询相关记忆。

        Args:
            query:          查询文本
            user_id:        学生唯一 ID
            memory_types:   过滤类型，可选 profile / episodic_memory / event_log / foresight
            top_k:          返回条数
            retrieve_method: keyword | vector | hybrid | rrf | agentic

        Returns:
            {"status": "ok", "result": {"memories": [...], "profiles": [...], ...}}
        """
        params: dict[str, Any] = {
            "user_id":         user_id,
            "query":           query,
            "top_k":           top_k,
            "retrieve_method": retrieve_method,
            "include_metadata": True,
        }
        if memory_types:
            params["memory_types"] = ",".join(memory_types)
        if group_ids:
            params["group_ids"] = ",".join(group_ids)

        return await self._get("/api/v0/memories/search", params)

    # ─────────────────────────────────────────
    # Internal HTTP helpers
    # ─────────────────────────────────────────

    async def _post(self, path: str, payload: dict) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=self._headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[EverMemOS] POST {path} 失败: HTTP {e.response.status_code} "
                f"body={e.response.text[:200]}"
            )
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            logger.warning(f"[EverMemOS] POST {path} 网络错误: {e}")
            return {"status": "error", "detail": str(e)}

    async def _get(self, path: str, params: dict) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"[EverMemOS] GET {path} 失败: HTTP {e.response.status_code} "
                f"body={e.response.text[:200]}"
            )
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            logger.warning(f"[EverMemOS] GET {path} 网络错误: {e}")
            return {"status": "error", "detail": str(e)}
