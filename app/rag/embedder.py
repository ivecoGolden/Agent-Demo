from openai import AsyncOpenAI
from app.core.config import settings

_embedder_instance = None


def get_aliyun_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = AliyunEmbedder()
    return _embedder_instance


class AliyunEmbedder:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=settings.ALI_LLM_BASE_URL,
            api_key=settings.ALI_LLM_KEY,
        )

    async def embed_texts(self, texts):
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                model=settings.EMBEDDING_MODEL_NAME,
                input=batch,
                dimensions=1024,
                encoding_format="float",
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
