"""
Textbook Knowledge Base Manager
================================

教材向量库管理器。负责将各科目的教材 PDF 一次性索引成向量库，
之后每次 explain 请求直接检索，无需重复索引。

索引路径（由 VectorIndexer 自动生成）：
    data/knowledge_bases/
    ├── english_tutorial.pdf        ← 源 PDF
    └── english_tutorial/           ← 向量库目录（自动生成）
        └── vector_store/
            ├── metadata.json       ← DenseRetriever 检测此文件判断是否已索引
            ├── embeddings.pkl      ← FAISS 不可用时的备用
            └── info.json

添加新教材：只需把 PDF 放入 data/knowledge_bases/ 并在 _SUBJECT_KB 里注册即可。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from src.logging import get_logger

logger = get_logger("KBManager")


# ── Subject → Textbook Config ─────────────────────────────────────────────────
# kb_name : 向量库目录名（= data/knowledge_bases/{kb_name}/）
# pdf     : PDF 文件名（位于 data/knowledge_bases/ 下）
# desc    : 日志友好名称
_SUBJECT_KB: dict[str, dict[str, str]] = {
    "english": {
        "kb_name": "english_tutorial",
        "pdf":     "english_tutorial.pdf",
        "desc":    "人教版八年级英语教材",
    },
    "biology": {
        "kb_name": "biology_tutorial",
        "pdf":     "biology_tutorial.pdf",
        "desc":    "人教版八年级生物教材",
    },
    "history": {
        "kb_name": "history_tutorial",
        "pdf":     "history_tutorial.pdf",
        "desc":    "人教版八年级下册历史教材",   # 已上传：下册
    },
    "chinese": {
        "kb_name": "chinese_tutorial",
        "pdf":     "chinese_tutorial.pdf",
        "desc":    "人教版八年级语文教材",
    },
}

# Per-subject asyncio.Lock：保证同一科目的首次索引不会被并发重复触发
_index_locks: dict[str, asyncio.Lock] = {}


def _get_lock(subject: str) -> asyncio.Lock:
    if subject not in _index_locks:
        _index_locks[subject] = asyncio.Lock()
    return _index_locks[subject]


# ── Manager ────────────────────────────────────────────────────────────────────

class TextbookKBManager:
    """
    教材向量库的懒加载管理器。

    典型用法：
        manager = TextbookKBManager(data_dir=Path("data"))

        # 确保索引存在（首次调用会触发 PDF → embed → FAISS 建库，约 1-3 分钟）
        kb_name = await manager.ensure_indexed("english")

        # 检索相关教材片段
        if kb_name:
            snippets = await manager.search(kb_name, "huge enormous vocabulary")
    """

    def __init__(self, data_dir: Path) -> None:
        self._kb_base = data_dir / "knowledge_bases"
        self._kb_base.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def supports_subject(self, subject: str) -> bool:
        """该科目是否有教材配置（不论 PDF 是否存在）。"""
        return subject in _SUBJECT_KB

    def get_textbook_desc(self, subject: str) -> str:
        """返回教材描述字符串，用于 prompt 中注明具体教材版本（如"人教版八年级下册历史教材"）。"""
        cfg = _SUBJECT_KB.get(subject.lower())
        return cfg["desc"] if cfg else ""

    def is_indexed(self, subject: str) -> bool:
        """快速判断：向量库是否已在磁盘上存在（不触发任何索引操作）。"""
        cfg = _SUBJECT_KB.get(subject)
        if not cfg:
            return False
        marker = self._kb_base / cfg["kb_name"] / "vector_store" / "metadata.json"
        return marker.exists()

    async def ensure_indexed(self, subject: str) -> Optional[str]:
        """
        确保教材向量库就绪。

        - 已索引 → 立即返回 kb_name（快速路径，仅文件检测）
        - 未索引 + PDF 存在 → 触发建库（慢，约 1-3 分钟），返回 kb_name
        - PDF 不存在 → 返回 None（调用方应降级为纯 LLM 解析）

        线程安全：同一科目同时只有一个协程在建库，其余等待完成后直接复用。
        """
        cfg = _SUBJECT_KB.get(subject)
        if not cfg:
            return None

        kb_name = cfg["kb_name"]
        pdf_path = self._kb_base / cfg["pdf"]

        # ── 快速路径：已有索引 ──────────────────────────────────────────────
        if self.is_indexed(subject):
            return kb_name

        # ── PDF 不存在 → 优雅降级 ──────────────────────────────────────────
        if not pdf_path.exists():
            logger.warning(
                f"[KBManager] 教材 PDF 未找到: {pdf_path.name}，"
                f"科目 '{subject}' 将回退为纯 LLM 解析"
            )
            return None

        # ── 慢路径：建库（加锁防并发重复建库）──────────────────────────────
        async with _get_lock(subject):
            # 取锁后再次检查（其他协程可能刚建完）
            if self.is_indexed(subject):
                return kb_name

            logger.info(
                f"[KBManager] 开始索引教材《{cfg['desc']}》: {pdf_path.name} "
                f"（首次请求，预计需 1-3 分钟）"
            )
            success = await self._build_index(kb_name, pdf_path)

        if success:
            logger.info(f"[KBManager] 教材索引完成: {kb_name}")
            return kb_name

        logger.error(f"[KBManager] 教材索引失败: {kb_name}")
        return None

    async def search(
        self,
        kb_name: str,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        """
        在向量库中检索与 query 最相关的教材片段。
        返回文本列表（按相关度排列），检索失败时返回空列表。
        """
        from src.services.rag.components.retrievers.dense import DenseRetriever

        retriever = DenseRetriever(kb_base_dir=str(self._kb_base), top_k=top_k)
        try:
            result = await retriever.process(query, kb_name)
            sources = result.get("results", [])
            return [
                s["content"]
                for s in sources
                if s.get("content", "").strip()
            ]
        except Exception as exc:
            logger.warning(f"[KBManager] 检索失败 ({kb_name}): {exc}")
            return []

    # ── Private ────────────────────────────────────────────────────────────────

    async def _build_index(self, kb_name: str, pdf_path: Path) -> bool:
        """
        Pipeline: PDFParser (PyMuPDF) → FixedSizeChunker → OpenAIEmbedder → VectorIndexer

        完全使用 src/services/rag/components/* 中已有组件组合而成，
        不依赖 llamaindex / lightrag / raganything 等重量级后端。

        Chunk 策略（教材 PDF）：
          - chunk_size=600 chars：够大能保留完整语义（一段课文/一个词汇表块），
                                  够小能保持检索精准度
          - chunk_overlap=80 chars：跨块保留上下文衔接
        """
        try:
            from src.services.rag.pipeline import RAGPipeline
            from src.services.rag.components.parsers.pdf import PDFParser
            from src.services.rag.components.chunkers.fixed import FixedSizeChunker
            from src.services.rag.components.embedders.openai import OpenAIEmbedder
            from src.services.rag.components.indexers.vector import VectorIndexer

            pipeline = (
                RAGPipeline("textbook_kb", kb_base_dir=str(self._kb_base))
                .parser(PDFParser(use_mineru=False))           # PyMuPDF fallback，无需 MinerU
                .chunker(FixedSizeChunker(chunk_size=600, chunk_overlap=80))
                .embedder(OpenAIEmbedder(batch_size=50))
                .indexer(VectorIndexer(kb_base_dir=str(self._kb_base)))
            )
            return await pipeline.initialize(kb_name, [str(pdf_path)])

        except Exception as exc:
            logger.error(f"[KBManager] 索引构建异常: {exc}", exc_info=True)
            return False
