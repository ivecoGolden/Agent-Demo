from typing import List
from app.memory.milvus_memory_handler import get_milvus_memory_handler
from app.rag.embedder import get_aliyun_embedder


class MemoryService:
    def __init__(self):
        self.milvus_handler = get_milvus_memory_handler()
        self.embedder = get_aliyun_embedder()

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        return await self.embedder.embed_texts(texts)

    @staticmethod
    def parse_memory_response(raw_results: List[dict]) -> List[dict]:
        return [
            {
                "timestamp": r["entity"]["timestamp"],
                "content": r["entity"]["content"],
                "category": r["entity"]["category"],
            }
            for r in raw_results
        ]

    async def search_user_memory(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[dict]:
        embedding = await self._embed([query])
        return self.milvus_handler.search_memory(
            user_id=user_id, embedding=embedding[0], top_k=top_k
        )

    async def search_user_memory_parsed(
        self, query: str, user_id: str, top_k: int = 5
    ) -> List[dict]:
        embedding = await self._embed([query])
        raw_results = self.milvus_handler.search_memory(
            user_id=user_id, embedding=embedding[0], top_k=top_k
        )
        return self.parse_memory_response(raw_results)

    def clear_user_memory(self, user_id: str) -> bool:
        """
        清除指定用户的所有记忆
        """
        return self.milvus_handler.delete_user_memory(user_id=user_id)

    async def add_user_memories(
        self,
        user_id: str,
        contents: List[str],
        categories: List[str],
        source: str = "chat",
    ) -> bool:
        """
        向用户的长期记忆中批量添加记录
        """
        embeddings = await self._embed(contents)
        return self.milvus_handler.insert_memory_bulk(
            user_id=user_id,
            embeddings=embeddings,
            contents=contents,
            categories=categories,
            source=source,
        )


_memory_service_instance: MemoryService = None


def get_memory_service() -> MemoryService:
    """获取内存服务实例"""
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    return _memory_service_instance
