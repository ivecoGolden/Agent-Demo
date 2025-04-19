from fastapi import APIRouter, Query
from typing import Optional
from pydantic import BaseModel
from app.rag.milvus_handler import MilvusHandler
from app.rag.retriever import Retriever
from app.core.startup import initialize_product_docs
from app.rag.embedder import get_aliyun_embedder

router = APIRouter(tags=["RAG 测试"])


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


handler = MilvusHandler()


@router.post("/rag/test")
async def rag_query(request: QueryRequest):
    retriever = Retriever(get_aliyun_embedder(), handler)

    chunks = await retriever.search(request.query, request.top_k)
    return {"query": request.query, "results": chunks}


@router.delete("/rag/collection")
async def delete_collection(
    collection_name: Optional[str] = Query(default=None, description="要删除的集合名")
):
    handler.delete_collection(collection_name)
    return {
        "message": f"Collection '{collection_name or handler.collection_name}' deleted successfully."
    }


@router.post("/rag/initialize")
async def initialize_docs():
    """手动触发产品说明文档的向量化处理"""
    await initialize_product_docs()
    return {"message": "产品文档已成功向量化"}


@router.get("/rag/memory")
async def search_user_memory(
    query: str = Query(..., description="用户查询内容"),
    user_id: str = Query(..., description="用户ID"),
):
    from app.memory.memory_service import get_memory_service

    service = get_memory_service()
    results = await service.search_user_memory(query=query, user_id=user_id)
    return {"query": query, "results": results}
