import os
# Force CPU-only mode by hiding any GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load environment variables for local development (if available)
from dotenv import load_dotenv
load_dotenv()

import torch
# Disable JIT profiling executor and MKLDNN to avoid backend conversion issues
torch._C._jit_set_profiling_executor(False)
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

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import BytesIO

import huggingface_hub

# # === Restore missing OfflineModeIsEnabled on hf_hub.utils ===
# try:
#     # import the new location...
#     from huggingface_hub.errors import OfflineModeIsEnabled
#     # ...and bind it back onto the old utils module
#     import huggingface_hub.utils as _hf_utils
#     _hf_utils.OfflineModeIsEnabled = OfflineModeIsEnabled
# except ImportError:
#     pass
#
# #### Monkey-patch HF downloads to drop unsupported kwargs ####
# if hasattr(huggingface_hub, "hf_hub_download"):
#     _orig_hf_hub_download = huggingface_hub.hf_hub_download
#     def _patched_hf_hub_download(*args, **kwargs):
#         # remove any kwargs the installed version doesn’t accept
#         kwargs.pop("url", None)
#         kwargs.pop("legacy_cache_layout", None)
#         return _orig_hf_hub_download(*args, **kwargs)
#     # override both entry-points
#     huggingface_hub.hf_hub_download = _patched_hf_hub_download
#     huggingface_hub.cached_download  = _patched_hf_hub_download

from sentence_transformers import SentenceTransformer
from transformers import LayoutLMTokenizerFast, LayoutLMModel

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

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

def extract_words_and_boxes_from_pages(pdf, start_page, end_page):
    """
    Flattens all words and their bboxes from pdf.pages[start_page:end_page].
    Returns:
      words: List[str], boxes: List[List[int]] = [x0, y0, x1, y1]
    """
    words, boxes = [], []
    for p in range(start_page, end_page):
        page = pdf.pages[p]
        for w in page.extract_words():
            words.append(w["text"])
            # pdfplumber uses keys x0, top, x1, bottom
            boxes.append([
                int(w["x0"]), int(w["top"]),
                int(w["x1"]), int(w["bottom"])
            ])
    return words, boxes

def pdf_section_to_html(filepath, start_page, end_page) -> str:
    """
    Extract the given page range as HTML bytes, then decode to str.
    """
    output = BytesIO()
    laparams = LAParams()
    with open(filepath, "rb") as fp:
        extract_text_to_fp(
            fp,
            output,
            laparams=laparams,
            output_type="html",
            page_numbers=list(range(start_page, end_page))
        )
    html_bytes = output.getvalue()
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # fallback if encoding differs
        return html_bytes.decode("latin-1")

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
    # def __init__(self, pinecone_index_name, model_name='all-MiniLM-L6-v2',
    #              chunk_size=500, chunk_overlap=50, chunks_path="chunks.pkl"):
    def __init__(self,
                 pinecone_index_name,
                 model_name='microsoft/layoutlm-base-uncased',
                  chunk_size=500,
                  chunk_overlap=50,
                  chunks_path="chunks.pkl"):
        # Initialize LayoutLM tokenizer + model on CPU
        self.tokenizer         = LayoutLMTokenizerFast.from_pretrained(model_name)
        #self.feature_extractor = LayoutLMFeatureExtractor(apply_ocr=False)
        self.model             = LayoutLMModel.from_pretrained(model_name).to('cpu')
        # for plain-text embeddings
        #self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_encoder  = SentenceTransformer('all-mpnet-base-v2')
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

    def ingest_document_with_layout(
        self,
        words: list[str],
        boxes: list[list[int]],
        source: str = "unknown",
        text: str | None = None,
        html: str | None = None
    ) -> None:
        """
        Ingest a single layout-aware document chunk:
          - words: list of word tokens
          - boxes: parallel list of [x0, y0, x1, y1] for each word
          - source: identifier for document/section
          - text: optional plain-text with \n\n preserved
          - html: optional HTML/Markdown rendition
        """
        # 1) Tokenize into LayoutLM inputs
        tok_out = self.tokenizer(
            words,
            return_tensors="pt",
            is_split_into_words=True,
            padding=True,
            truncation=True
        )

        # 2) Align each token back to its original bbox
        word_id_list = tok_out.word_ids(batch_index=0)
        aligned_boxes: list[list[int]] = []
        for word_id in word_id_list:
            if word_id is None:
                aligned_boxes.append([0, 0, 0, 0])
            else:
                aligned_boxes.append(boxes[word_id])

        # 3) Build the tensor for LayoutLM’s bbox input
        bbox_tensor = torch.tensor([aligned_boxes], dtype=torch.long)

        # 4) Forward through LayoutLM
        outputs = self.model(
            input_ids      = tok_out.input_ids,
            attention_mask = tok_out.attention_mask,
            bbox           = bbox_tensor
        )
        emb = outputs.pooler_output[0].detach().cpu().numpy().astype('float32')

        # 5) Prepare your chunk’s text/html and metadata
        idx        = len(self.chunks)
        chunk_text = text if text is not None else " ".join(words)
        chunk_html = html  # may be None
        self.chunks.append({
            "text":   chunk_text,
            "html":   chunk_html,
            "source": source
        })

        metadata = {"source": source}

        # 6) Upsert into Pinecone with both text & html in metadata
        self.index.upsert(vectors=[(str(idx), emb.tolist(), metadata)])

        # 7) Rebuild your BM25 index on the plain-text corpus
        self._build_sparse_index()

    def query(self,
              query_text: str,
              top_k: int = 3,
              alpha: float = 0.5,
              dense_k: int = 10,
              sparse_k: int = 10) -> list[dict]:
        """
        Perform a hybrid dense + sparse retrieval, then merge results.
        Each result dict contains: id, text, html, source, score.
        """
        # 1) Embed the query with your text_encoder
        q_emb = self.text_encoder.encode([query_text])[0].astype('float32')

        # 2) Dense (semantic) retrieval via Pinecone
        dense_matches: list[dict] = []
        if dense_k > 0:
            dense_resp = self.index.query(
                vector=q_emb.tolist(),
                top_k=dense_k,
                include_metadata=True
            )
            for m in dense_resp["matches"]:
                idx = int(m["id"])
                dense_matches.append({
                    "id":     m["id"],
                    "score":  m["score"],
                    "text":   self.chunks[idx]["text"],
                    "html":   self.chunks[idx].get("html"),
                    "source": self.chunks[idx]["source"]
                })

        # 3) Sparse (keyword) retrieval via BM25
        sparse_matches: list[dict] = []
        if sparse_k > 0 and BM25Okapi and getattr(self, "bm25", None):
            tokens = query_text.split()
            scores = self.bm25.get_scores(tokens)
            top_sparse = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )[:sparse_k]

            for idx, score in top_sparse:
                sparse_matches.append({
                    "id":     str(idx),
                    "score":  score,
                    "text":   self.chunks[idx]["text"],
                    "html":   self.chunks[idx].get("html"),
                    "source": self.chunks[idx]["source"]
                })

        # 4) Normalize scores to [0,1]
        def normalize(matches: list[dict]) -> dict[str, float]:
            if not matches:
                return {}
            vals = [m["score"] for m in matches]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-6:
                return {m["id"]: 1.0 for m in matches}
            return {m["id"]: (m["score"] - lo) / (hi - lo) for m in matches}

        d_norm = normalize(dense_matches)
        s_norm = normalize(sparse_matches)

        # 5) Merge by weighted sum
        merged: dict[str, dict] = {}
        for m in dense_matches + sparse_matches:
            d_sc = d_norm.get(m["id"], 0.0)
            s_sc = s_norm.get(m["id"], 0.0)
            combined = alpha * d_sc + (1 - alpha) * s_sc
            if (m["id"] not in merged) or (merged[m["id"]]["score"] < combined):
                merged[m["id"]] = {
                    "id":     m["id"],
                    "text":   m["text"],
                    "html":   m.get("html"),
                    "source": m["source"],
                    "score":  combined
                }

        # 6) Return the top_k results
        results = sorted(
            merged.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        return results

        # — normalize scores to [0,1] —
        def normalize(xs):
            if not xs: return {}
            vals = [m["score"] for m in xs]
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-6:
                return {m["id"]: 1.0 for m in xs}
            return {m["id"]: (m["score"] - lo) / (hi - lo) for m in xs}

        d_norm = normalize(dense_matches)
        s_norm = normalize(sparse_matches)

        # — merge by weighted sum —
        merged = {}
        for m in dense_matches + sparse_matches:
            d_sc = d_norm.get(m["id"], 0.0)
            s_sc = s_norm.get(m["id"], 0.0)
            combined = alpha * d_sc + (1 - alpha) * s_sc
            # keep highest combined score for each chunk
            if m["id"] not in merged or merged[m["id"]]["score"] < combined:
                merged[m["id"]] = {
                    "text": m["text"],
                    "source": m["source"],
                    "score": combined
                }

        # — sort and return top_k —
        results = sorted(merged.values(),
                         key=lambda x: x["score"],
                         reverse=True)[:top_k]
        return results

    def save_index(self, chunks_path=None):
        if chunks_path is not None:
            self.chunks_path = chunks_path
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        """
        Load any previously‐saved chunks from disk, then rebuild the sparse (BM25) index.
        """
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                # failed to unpickle? start fresh
                self.chunks = []
        else:
            self.chunks = []

        # If BM25 is available and we have chunks, rebuild sparse index
        if BM25Okapi and self.chunks:
            self._build_sparse_index()

    def _build_sparse_index(self):
        # no chunks → skip
        if not self.chunks:
            self.bm25 = None
            return
        # build BM25
        corpus = [c["text"].split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

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

        # If we have sources, enumerate them for the model
        if valid_sources:
            # Build a numbered list
            src_list = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(valid_sources))
            messages.append({
                "role": "system",
                "content": (
                    "Here are the available sources to cite:\n" + src_list +
                    "\n\nAfter each sentence in your answer, append a superscript citation like ^[1] "
                    "to indicate which source you used."
                )
            })

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
        rag = st.session_state["rag"]
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

                    # === Text-only PDF ingestion ===
                    with pdfplumber.open(filepath) as pdf:
                        if sections:
                            logger.info(f"PDF {filename} has {len(sections)} outline sections")
                            for title, start_page, end_page in sections:
                                # 1) Log what pages this section covers
                                logger.debug(
                                    f"Section '{title}' spans pages [{start_page}..{end_page - 1}]"
                                )
                                # 2) Pull raw text from that page range
                                page_texts = []
                                for p in range(start_page, end_page):
                                    if p < len(pdf.pages):
                                        txt = pdf.pages[p].extract_text() or ""
                                        if txt.strip():
                                            page_texts.append(txt)
                                # 3) If truly empty, warn once
                                if not page_texts:
                                    logger.warning(
                                        f"Section '{title}' (pages {start_page}–{end_page-1}) has no text; skipping."
                                    )
                                    continue

                                # 4) Extract layout info and also preserve the raw paragraphs
                                cleaned_texts = remove_headers_and_footers(page_texts)
                                raw_text = "\n\n".join(cleaned_texts)
                                words, boxes = extract_words_and_boxes_from_pages(
                                    pdf, start_page, end_page
                                )
                                if not words:
                                    logger.warning(
                                        f"Section '{title}' empty after layout extraction; skipping."
                                    )
                                    continue
                                meta_str = f"{filename} | Section: {title}"
                                html = pdf_section_to_html(filepath, start_page, end_page)
                                st.session_state["rag"].ingest_document_with_layout(
                                    words,
                                    boxes,
                                    source=meta_str,
                                    text=raw_text,
                                    html=html
                                )
                                time.sleep(0.3)

                        else:
                            logger.info(f"No outline for {filename}; ingesting by page")
                            for page_idx, page in enumerate(pdf.pages, start=1):
                                # 1) Pull raw text for this single page
                                page_text = page.extract_text() or ""
                                if not page_text.strip():
                                    continue
                                # 2) Preserve it as raw_text and get layout tokens
                                raw_text = page_text
                                words, boxes = extract_words_and_boxes_from_pages(
                                    pdf, page_idx-1, page_idx
                                )
                                if not words:
                                    continue
                                meta_str = f"{filename} | Page: {page_idx}"
                                # 3) Convert this page to HTML
                                html = pdf_section_to_html(filepath, page_idx-1, page_idx)
                                # 4) Ingest with both text and html
                                st.session_state["rag"].ingest_document_with_layout(
                                    words,
                                    boxes,
                                    source=meta_str,
                                    text=raw_text,
                                    html=html
                                )
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
            # Render one collapsible section per citation
            if "citations" in msg and msg["citations"]:
                for citation in msg["citations"]:
                    # if you ingested real HTML, use it; otherwise fall back to plain text
                    content = citation.get("html") or citation["content"]
                    html += (
                        f'<div class="chat-bubble citation-bubble" '
                          'style="border-radius: 15px; padding: 10px; background-color: #f8f8f8;">'
                        f'<details>'
                          f'<summary style="font-size: 14px; color: #666;">'
                            f'Citation: {format_citation(citation["header"])}'
                          f'</summary>'
                          f'<div style="white-space: pre-wrap; font-size: 13px; padding: 8px;">'
                            f'{content}'
                          f'</div>'
                        f'</details>'
                        f'</div>'
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
