class Retriever:
    def __init__(self, embedder, milvus_handler):
        self.embedder = embedder
        self.milvus_handler = milvus_handler

    async def search(self, query: str, top_k: int = 5):
        vectorized_query = (await self.embedder.embed_texts([query]))[0]
        matched_chunks = self.milvus_handler.search(vectorized_query, top_k)
        return matched_chunks
