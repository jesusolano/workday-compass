import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class RAG:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=500, chunk_overlap=50, index_path="faiss.index", chunks_path="chunks.pkl"):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = None
        self.embeddings = None
        self.chunks = []
        self.load_index()  # Load persisted data if available

    def _split_text(self, text):
        text = text.replace('\n', ' ')
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    def ingest_document(self, text, source="unknown"):
        new_chunks = self._split_text(text)
        for chunk in new_chunks:
            self.chunks.append({"text": chunk, "source": source})
        embeddings = self.model.encode(new_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        else:
            self.embeddings = embeddings
        if self.index is None:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def query(self, query_text, top_k=3):
        query_embedding = self.model.encode([query_text])
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def save_index(self, index_path=None, chunks_path=None):
        if index_path is not None:
            self.index_path = index_path
        if chunks_path is not None:
            self.chunks_path = chunks_path
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            # If chunks are stored as strings from previous versions, convert them.
            if self.chunks and isinstance(self.chunks[0], str):
                self.chunks = [{"text": chunk, "source": "unknown"} for chunk in self.chunks]
        else:
            # No persisted data found, start with empty values.
            self.index = None
            self.embeddings = None
            self.chunks = []
