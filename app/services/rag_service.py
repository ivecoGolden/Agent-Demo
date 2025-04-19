from typing import List

# 引入所需的类
from app.rag.embedder import AliyunEmbedder
from app.rag.milvus_handler import MilvusHandler
from app.rag.retriever import Retriever


class RAGService:
    def __init__(self):
        # 初始化嵌入器、Milvus处理器和检索器
        self.embedder = AliyunEmbedder()  # 创建嵌入器实例
        self.milvus_handler = MilvusHandler()  # 创建Milvus处理器实例
        self.retriever = Retriever(self.embedder, self.milvus_handler)  # 创建检索器实例

    async def query(self, question: str, top_k: int = 2) -> List[str]:
        # 将问题转化为嵌入向量
        embedding = await self.embedder.embed_texts([question])  # 异步获取嵌入向量
        # 从Milvus中检索最匹配的文档
        matched_chunks = self.milvus_handler.search(
            embedding[0], top_k
        )  # 同步检索匹配的文档
        return matched_chunks  # 返回匹配到的文档片段
