import os
# Force CPU-only mode by hiding any GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load environment variables for local development (if available)
from dotenv import load_dotenv
load_dotenv()

import torch
# Disable MKLDNN to prevent unsupported operations during device conversion
torch.backends.mkldnn.enabled = False

import time
import re
import pickle
import hashlib
import streamlit as st
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import pdfplumber
from PyPDF2 import PdfReader

# Global paths and threshold
SIMILARITY_THRESHOLD = 0.55
INGESTED_FILES_PATH = "ingested_files.txt"
PROCESSING_FILES_PATH = "processing_files.txt"

# Create and configure a logger for this module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# If the logger doesn't already have handlers, add one for file logging.
if not logger.handlers:
    file_handler = logging.FileHandler("document_ingestion.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# -------------------- Helper Functions --------------------
def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_file_set(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return set(line.strip() for line in f)
    return set()

def append_to_file(filepath, identifier):
    with open(filepath, "a") as f:
        f.write(identifier + "\n")

def remove_from_file(filepath, identifier):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        lines = f.readlines()
    with open(filepath, "w") as f:
        for line in lines:
            if line.strip() != identifier:
                f.write(line)

def load_ingested_files():
    return load_file_set(INGESTED_FILES_PATH)

def save_ingested_files(file_set):
    with open(INGESTED_FILES_PATH, "w") as f:
        for item in file_set:
            f.write(item + "\n")

def dynamic_chunk_text(text, max_words=700):
    """
    Keeps paragraphs intact and splits them only if they exceed max_words.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    current_words = 0
    for para in paragraphs:
        para_word_count = len(para.split())
        if para_word_count >= max_words:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_words = 0
            words = para.split()
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i+max_words])
                chunks.append(chunk)
            continue
        if current_words + para_word_count > max_words:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            current_words = para_word_count
        else:
            current_chunk += para + "\n\n"
            current_words += para_word_count
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def flatten_outlines(outline_list, pdf_reader, results=None, parent_title=""):
    if results is None:
        results = []
    for item in outline_list:
        if isinstance(item, list):
            flatten_outlines(item, pdf_reader, results, parent_title)
        else:
            title = getattr(item, "title", "Untitled")
            full_title = f"{parent_title} > {title}" if parent_title else title
            try:
                page_number = pdf_reader.get_destination_page_number(item)
            except:
                page_number = 0
            results.append((full_title, page_number))
            children = getattr(item, "children", None)
            if children:
                if callable(children):
                    children = children()
                if isinstance(children, list):
                    flatten_outlines(children, pdf_reader, results, full_title)
    return results

def get_outline_sections(pdf_reader):
    if not hasattr(pdf_reader, "outline"):
        return []
    outline_obj = pdf_reader.outline
    if callable(outline_obj):
        outline_obj = outline_obj()
    if not outline_obj:
        return []
    raw_list = flatten_outlines(outline_obj, pdf_reader)
    raw_list.sort(key=lambda x: x[1])
    sections = []
    for i, (title, start_page) in enumerate(raw_list):
        end_page = raw_list[i+1][1] if i < len(raw_list)-1 else len(pdf_reader.pages)
        sections.append((title, start_page, end_page))
    return sections

def remove_common_footer(page_texts, min_percentage=0.5, max_length=None):
    footer_candidates = []
    for text in page_texts:
        lines = text.strip().splitlines()
        if lines:
            footer_candidates.append(lines[-1].strip())
    if not footer_candidates:
        return page_texts
    counter = Counter(footer_candidates)
    common_footer, count = counter.most_common(1)[0]
    if count / len(footer_candidates) >= min_percentage:
        if max_length is None or len(common_footer) <= max_length:
            new_texts = []
            for text in page_texts:
                lines = text.splitlines()
                if lines and lines[-1].strip() == common_footer:
                    lines = lines[:-1]
                new_texts.append("\n".join(lines))
            return new_texts
    return page_texts

def remove_common_header(page_texts, min_percentage=0.5, max_length=None):
    header_candidates = []
    for text in page_texts:
        lines = text.strip().splitlines()
        if lines:
            header_candidates.append(lines[0].strip())
    if not header_candidates:
        return page_texts
    counter = Counter(header_candidates)
    common_header, count = counter.most_common(1)[0]
    if count / len(header_candidates) >= min_percentage:
        if max_length is None or len(common_header) <= max_length:
            new_texts = []
            for text in page_texts:
                lines = text.splitlines()
                if lines and lines[0].strip() == common_header:
                    lines = lines[1:]
                new_texts.append("\n".join(lines))
            return new_texts
    return page_texts

def remove_headers_and_footers(page_texts):
    texts_no_header = remove_common_header(page_texts, min_percentage=0.5, max_length=None)
    texts_no_header_footer = remove_common_footer(texts_no_header, min_percentage=0.5, max_length=None)
    return texts_no_header_footer

def format_citation(source):
    parts = [part.strip() for part in source.split("|")]
    parts = [part for part in parts if not re.match(r'^chunk:', part, re.IGNORECASE)]
    return " | ".join(parts)

def smart_format_text(text):
    """
    Converts raw text into HTML.
    """
    lines = text.splitlines()
    formatted = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted += "<br>"
        elif stripped[-1] in ".?!;:":
            formatted += stripped + "<br>"
        else:
            formatted += stripped + " "
    return formatted

# -------------------- RAG and Generator Classes --------------------
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

class RAG:
    def __init__(self, pinecone_index_name, model_name='all-MiniLM-L6-v2',
                 chunk_size=500, chunk_overlap=50, chunks_path="chunks.pkl"):
        # Do not pass an explicit device so that the model auto-selects CPU.
        # With CUDA hidden, it should run on CPU.
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks_path = chunks_path
        self.chunks = []
        self.pinecone_index_name = pinecone_index_name
        # Retrieve the Pinecone API key from Streamlit secrets
        self.index = Pinecone(api_key=st.secrets["PINECONE_API_KEY"]).Index(pinecone_index_name)
        self.load_index()

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

    def ingest_document(self, text, source="unknown", pre_split=False):
        if pre_split:
            new_chunks = [text]
        else:
            new_chunks = self._split_text(text)
        start_index = len(self.chunks)
        for chunk in new_chunks:
            self.chunks.append({"text": chunk, "source": source})
        embeddings = self.model.encode(new_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        vectors = []
        for i, emb in enumerate(embeddings):
            vector_id = str(start_index + i)
            metadata = {"text": new_chunks[i], "source": source}
            vectors.append((vector_id, emb.tolist(), metadata))
        self.index.upsert(vectors=vectors)

    def query(self, query_text, top_k=3):
        query_embedding = self.model.encode([query_text]).tolist()[0]
        result = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = result.get("matches", [])
        results = []
        for match in matches:
            meta = dict(match.get("metadata", {}))
            meta["score"] = match.get("score", 0)
            results.append(meta)
        return results

    def save_index(self, chunks_path=None):
        if chunks_path is not None:
            self.chunks_path = chunks_path
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            if self.chunks and isinstance(self.chunks[0], str):
                self.chunks = [{"text": c, "source": "unknown"} for c in self.chunks]
        else:
            self.chunks = []

class Generator:
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=2048):
        import openai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Retrieve OpenAI API key from Streamlit secrets
        openai.api_key = st.secrets["OPENAI_API_KEY"]

    def generate_answer(self, history, context, valid_sources):
        import openai
        messages = []
        system_prompt = (
            "You are a highly knowledgeable assistant. Answer the user's question based solely on the provided context "
            "and conversation history. Your response should be detailed and include citations where applicable."
        )
        messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response["choices"][0]["message"]["content"]

# -------------------- Document Ingestion --------------------
def ingest_documents():
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        st.warning(f"Directory '{docs_dir}' not found.")
        logger.warning(f"Directory '{docs_dir}' not found.")
        return

    ingested_files = load_ingested_files()
    processing_files = load_file_set(PROCESSING_FILES_PATH)
    new_files_found = False
    progress_placeholder = st.empty()

    logger.info("Starting document ingestion process...")
    with st.spinner("Ingesting new documents..."):
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            if not os.path.isfile(filepath):
                continue
            file_hash = compute_file_hash(filepath)
            logger.debug(f"Found file: {filename} with hash: {file_hash}.")
            if file_hash in ingested_files:
                logger.info(f"Skipping already ingested file: {filename}.")
                continue
            if file_hash in processing_files:
                logger.info(f"File {filename} is in processing queue, removing previous entry.")
                remove_from_file(PROCESSING_FILES_PATH, file_hash)
            append_to_file(PROCESSING_FILES_PATH, file_hash)
            new_files_found = True

            try:
                if filename.lower().endswith(".txt"):
                    progress_placeholder.text(f"Ingesting {filename}...")
                    logger.info(f"Ingesting text file: {filename}.")
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                    chunks = dynamic_chunk_text(text, max_words=700)
                    for i, chunk in enumerate(chunks):
                        meta_str = f"{filename} | Section: Full Document | Chunk: {i+1}/{len(chunks)}"
                        logger.debug(f"Ingesting chunk {i+1} of {len(chunks)} from text file: {filename}.")
                        st.session_state["rag"].ingest_document(chunk, source=meta_str, pre_split=True)
                        time.sleep(0.3)
                    progress_placeholder.text(f"Ingested {filename}")
                    logger.info(f"Successfully ingested text file: {filename}.")
                    time.sleep(0.5)

                elif filename.lower().endswith(".pdf"):
                    progress_placeholder.text(f"Ingesting {filename}...")
                    logger.info(f"Ingesting PDF file: {filename}.")
                    with open(filepath, "rb") as f:
                        pdf_reader = PdfReader(f)
                        sections = get_outline_sections(pdf_reader)

                    with pdfplumber.open(filepath) as pdf:
                        if sections:
                            logger.info(f"PDF file {filename} has outline sections. Total sections: {len(sections)}")
                            for title, start_page, end_page in sections:
                                page_texts = []
                                for i in range(start_page, end_page):
                                    if i < len(pdf.pages):
                                        page = pdf.pages[i]
                                        page_text = page.extract_text() or ""
                                        if not page_text.strip():
                                            continue
                                        page_texts.append(page_text)
                                if not page_texts:
                                    logger.warning(f"Section '{title}' in {filename} is empty, skipping.")
                                    continue
                                page_texts = remove_headers_and_footers(page_texts)
                                section_text = "\n\n".join(page_texts)
                                chunks = dynamic_chunk_text(section_text, max_words=700)
                                for i, chunk in enumerate(chunks):
                                    meta_str = f"{filename} | Section: {title} | Chunk: {i+1}/{len(chunks)}"
                                    logger.debug(f"Ingesting chunk {i+1} of {len(chunks)} from section '{title}' in PDF file: {filename}.")
                                    st.session_state["rag"].ingest_document(chunk, source=meta_str, pre_split=True)
                                    time.sleep(0.3)
                        else:
                            logger.info(f"PDF file {filename} does not have outline sections, ingesting by pages.")
                            for page_idx, page in enumerate(pdf.pages):
                                page_text = page.extract_text() or ""
                                if not page_text.strip():
                                    continue
                                page_text = remove_headers_and_footers([page_text])[0]
                                page_chunks = dynamic_chunk_text(page_text, max_words=700)
                                for i, chunk in enumerate(page_chunks):
                                    meta_str = f"{filename} | Page: {page_idx + 1} | Chunk: {i+1}/{len(page_chunks)}"
                                    logger.debug(f"Ingesting chunk {i+1} of {len(page_chunks)} from page {page_idx+1} in PDF file: {filename}.")
                                    st.session_state["rag"].ingest_document(chunk, source=meta_str, pre_split=True)
                                    time.sleep(0.3)
                    progress_placeholder.text(f"Ingested {filename}")
                    logger.info(f"Successfully ingested PDF file: {filename}.")
                    time.sleep(0.5)

                else:
                    st.warning(f"Unsupported file type: {filename}")
                    logger.warning(f"Unsupported file type for file: {filename}.")

                append_to_file(INGESTED_FILES_PATH, file_hash)
                remove_from_file(PROCESSING_FILES_PATH, file_hash)
                logger.info(f"File {filename} ingestion completed and updated in ingested list.")
            except Exception as e:
                st.error(f"Error ingesting {filename}: {e}")
                logger.error(f"Error ingesting {filename}: {e}", exc_info=True)
        if new_files_found:
            progress_placeholder.text("")
            st.session_state["rag"].save_index()
            st.success("New documents ingested! Index updated!")
            logger.info("Document ingestion process completed successfully. Index updated.")
        else:
            logger.info("No new documents found for ingestion.")

# -------------------- Conversation Functions --------------------
def get_conversation_html():
    active_thread = get_active_thread()
    if not active_thread:
        return ""
    html = '<div class="chat-container">'
    for msg in active_thread["history"]:
        if msg["role"] == "user":
            html += (
                f'<div class="chat-bubble user-bubble">'
                f'<strong>User:</strong> {msg["content"]}'
                f'</div><div class="clearfix"></div>'
            )
        else:
            html += (
                f'<div class="chat-bubble assistant-bubble">'
                f'<strong>Assistant:</strong> {msg["content"]}'
                f'</div>'
            )
            if "citation" in msg and msg["citation"]:
                citation = msg["citation"]
                formatted_content = smart_format_text(citation["content"])
                html += (
                    f'<div class="chat-bubble citation-bubble" style="border-radius: 15px; padding: 10px; background-color: #f8f8f8;">'
                    f'<details><summary style="font-size: 14px; color: #666;">Top Citation: {format_citation(citation["header"])}</summary>'
                    f'<div style="font-size: 13px; padding: 8px;">{formatted_content}</div>'
                    f'</details></div>'
                )
        html += '<div class="clearfix"></div>'
    html += "</div>"
    return html

def create_new_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_thread = {"title": f"Conversation {timestamp}", "history": []}
    st.session_state.setdefault("conversation_threads", []).append(new_thread)
    return len(st.session_state["conversation_threads"]) - 1

def get_active_thread():
    idx = st.session_state.get("active_conversation_index", None)
    if idx is not None and idx < len(st.session_state.get("conversation_threads", [])):
        return st.session_state["conversation_threads"][idx]
    return None

def rename_conversation_auto(thread):
    if thread["title"].startswith("Conversation "):
        for msg in thread["history"]:
            if msg["role"] == "user" and msg["content"].strip():
                text = msg["content"].strip()
                short_title = text[:30] + "..." if len(text) > 30 else text
                thread["title"] = short_title
                return
        thread["title"] = "Untitled conversation"

def handle_delete_conversation():
    delete_choice = st.session_state.get("delete_convo", "")
    if not delete_choice:
        return
    for i, thread in enumerate(st.session_state.get("conversation_threads", [])):
        if thread["title"] == delete_choice:
            del st.session_state["conversation_threads"][i]
            if st.session_state.get("active_conversation_index") == i:
                st.session_state["active_conversation_index"] = 0 if st.session_state["conversation_threads"] else None
            elif (st.session_state.get("active_conversation_index") is not None and
                  st.session_state["active_conversation_index"] > i):
                st.session_state["active_conversation_index"] -= 1
            break
    st.session_state["rerun_trigger"] = not st.session_state.get("rerun_trigger", False)
