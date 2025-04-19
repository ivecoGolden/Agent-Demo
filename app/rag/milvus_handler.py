from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from pymilvus import utility
from typing import List


class MilvusHandler:
    def __init__(self, collection_name: str = "rag_documents"):
        connections.connect("default", host="localhost", port="19530")
        self.collection_name = collection_name
        self._create_collection()

    def _create_collection(self):
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
        )
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024
        )
        text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)

        schema = CollectionSchema(
            fields=[id_field, embedding_field, text_field],
            description="Collection for RAG documents",
        )

        if not utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name, schema=schema)
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            }
            self.collection.create_index(
                field_name="embedding", index_params=index_params
            )
        else:
            self.collection = Collection(name=self.collection_name)

        self.collection.load()

    def insert(self, vector: List[float], text: str):
        if utility.load_state(self.collection_name) != "Loaded":
            self.collection.load()
        data = [[vector], [text]]
        self.collection.insert(data=data, fields=["embedding", "text"])

    def insert_batch(self, vectors: List[List[float]], texts: List[str]):
        if utility.load_state(self.collection_name) != "Loaded":
            self.collection.load()
        assert len(vectors) == len(texts), "❌ 向量与文本数量不一致"
        for vec in vectors:
            assert len(vec) == 1024, f"❌ 向量维度错误，应为 1024，实际为: {len(vec)}"
        data = [vectors, texts]
        self.collection.insert(data=data, fields=["embedding", "text"])

    def search(self, vector: List[float], top_k: int) -> List[str]:
        if utility.load_state(self.collection_name) != "Loaded":
            self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self.collection.search(
            [vector], "embedding", search_params, limit=top_k, output_fields=["text"]
        )
        return [result.entity.get("text") for result in results[0]]

    def delete_collection(self, collection_name: str = None):
        """
        删除指定的 Collection。如果未指定，默认删除当前 handler 使用的 collection。
        """
        name = collection_name or self.collection_name
        if utility.has_collection(name):
            utility.drop_collection(name)
