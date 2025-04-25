from langsmith import traceable
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    connections,
    utility,
)
from typing import List, Dict
from app.core.config import settings
import time


class MilvusMemoryHandler:
    def __init__(self, collection_name="user_memory"):
        connections.connect("default", host=settings.MILVUS_HOST, port="19530")
        self.collection_name = collection_name
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self._load_collection_if_needed()
        else:
            self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="user_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_partition_key=True,
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="User memory store")
        self.collection = Collection(name=self.collection_name, schema=schema)
        self.collection.create_index(
            "embedding",
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        self.collection.load()

    def _load_collection_if_needed(self):
        if utility.load_state(self.collection_name) != "Loaded":
            self.collection.load()

    def insert_memory_bulk(
        self,
        user_id: str,
        embeddings: List[List[float]],
        contents: List[str],
        categories: List[str],
        source: str = "chat",
    ) -> None:
        count = len(embeddings)
        timestamps = [int(time.time())] * count
        user_ids = [user_id] * count
        sources = [source] * count

        data = [
            user_ids,
            embeddings,
            contents,
            categories,
            sources,
            timestamps,
        ]
        self.collection.insert(
            data=data,
            fields=[
                "user_id",
                "embedding",
                "content",
                "category",
                "source",
                "timestamp",
            ],
        )

    @traceable(run_type="retriever")
    def search_memory(
        self, user_id: str, embedding: List[float], top_k=5
    ) -> List[Dict]:
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "category", "source", "timestamp"],
        )
        hits = results[0] if results else []
        return [hit.entity.to_dict() for hit in hits]

    def delete_user_memory(self, user_id: str):
        expr = f'user_id == "{user_id}"'
        self.collection.delete(expr)


_milvus_memory_handler: MilvusMemoryHandler | None = None


def get_milvus_memory_handler() -> MilvusMemoryHandler:
    global _milvus_memory_handler
    if _milvus_memory_handler is None:
        _milvus_memory_handler = MilvusMemoryHandler()
    return _milvus_memory_handler
