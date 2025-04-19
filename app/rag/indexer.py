from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Indexer:

    def __init__(self, embedder, milvus_handler):
        self.embedder = embedder
        self.milvus_handler = milvus_handler
        self.text_splitter_md = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "一级标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ]
        )
        self.text_splitter_txt = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50
        )

    def split_markdown_text(self, text):
        header_chunks = self.text_splitter_md.split_text(text)
        all_chunks = []
        for header_chunk in header_chunks:
            inner_chunks = self.text_splitter_txt.split_text(header_chunk.page_content)
            all_chunks.extend(inner_chunks)
        return all_chunks

    async def index_document(self, text, file_type="txt"):
        if file_type == "md":
            chunks = self.split_markdown_text(text)
        else:
            chunks = self.text_splitter_txt.split_text(text)

        vectors = await self.embedder.embed_texts(chunks)
        texts = [chunk for chunk in chunks if chunk.strip()]  # 检查块是否不为空

        self.milvus_handler.insert_batch(vectors, texts)
