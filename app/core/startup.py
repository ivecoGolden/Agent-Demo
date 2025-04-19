import os
import logging
from app.rag.embedder import AliyunEmbedder
from app.rag.milvus_handler import MilvusHandler
from app.rag.indexer import Indexer

logging.basicConfig(level=logging.INFO)


async def initialize_product_docs():
    file_path = "docs/product.md"
    if not os.path.exists(file_path):
        logging.error(f"文件未找到: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        logging.info("成功读取产品文档。")

    embedder = AliyunEmbedder()
    embeddings = await embedder.embed_texts([content])
    logging.info("成功嵌入产品文档。")

    milvus_handler = MilvusHandler()
    indexer = Indexer(embedder, milvus_handler)
    await indexer.index_document(content, file_type="md")
    logging.info("成功索引产品文档。")
