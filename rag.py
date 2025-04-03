import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pinecone import Pinecone

class RAG:
    def __init__(self, pinecone_index_name, model_name='all-MiniLM-L6-v2',
                 chunk_size=500, chunk_overlap=50, chunks_path="chunks.pkl"):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks_path = chunks_path
        self.chunks = []
        self.pinecone_index_name = pinecone_index_name
        # Initialize Pinecone index using the new API.
        # Assumes PINECONE_API_KEY is set in the environment.
        self.index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(pinecone_index_name)
        self.load_index()  # Load persisted chunks if available

    def _split_text(self, text):
        # Default splitting method (by words) if text is not pre-chunked.
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

    def ingest_document(self, text, source="unknown", pre_split=False):
        """
        Ingest a document by splitting it into chunks, encoding them,
        and upserting to the Pinecone index.
        If pre_split is True, it assumes the text is already a single chunk.
        """
        if pre_split:
            new_chunks = [text]
        else:
            new_chunks = self._split_text(text)
        start_index = len(self.chunks)
        # Append new chunks locally (for citation metadata)
        for chunk in new_chunks:
            self.chunks.append({"text": chunk, "source": source})
        # Encode new chunks and ensure float32 format
        embeddings = self.model.encode(new_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        # Prepare vectors for upsert: each is a tuple (id, vector, metadata)
        vectors = []
        for i, emb in enumerate(embeddings):
            vector_id = str(start_index + i)
            metadata = {"text": new_chunks[i], "source": source}
            vectors.append((vector_id, emb.tolist(), metadata))
        # Upsert the new vectors to the Pinecone index
        self.index.upsert(vectors=vectors)

    def query(self, query_text, top_k=3):
        # Compute the query embedding and convert to list
        query_embedding = self.model.encode([query_text]).tolist()[0]
        # Query the Pinecone index
        result = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = result.get("matches", [])
        # Extract metadata and include the score in the returned dictionary
        results = []
        for match in matches:
            match_metadata = dict(match.get("metadata", {}))
            match_metadata["score"] = match.get("score", 0)
            results.append(match_metadata)
        return results

    def save_index(self, chunks_path=None):
        if chunks_path is not None:
            self.chunks_path = chunks_path
        # Persist the local chunks list (for citation/reference purposes)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            # If chunks were stored as strings in a previous version, convert them.
            if self.chunks and isinstance(self.chunks[0], str):
                self.chunks = [{"text": chunk, "source": "unknown"} for chunk in self.chunks]
        else:
            self.chunks = []
