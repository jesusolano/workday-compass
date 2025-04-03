import os
import time
import streamlit as st
from datetime import datetime
import numpy as np
import asyncio
import shutil
from dotenv import load_dotenv
import hashlib
from PyPDF2 import PdfReader
import logging
import sys
import pickle
import openai

# -------------------- Setup and Environment --------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

load_dotenv()

# Ensure required environment variables are set:
# - PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY, etc.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global thresholds
SIMILARITY_THRESHOLD = 0.55  # Global similarity threshold

# File paths for ingestion persistence.
INGESTED_FILES_PATH = "ingested_files.txt"
PROCESSING_FILES_PATH = "processing_files.txt"

# -------------------- RAG Class (Retrieval-Augmented Generation) --------------------
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

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

# -------------------- Generator Class --------------------
class Generator:
    def __init__(self, model="gpt-4.5-preview", temperature=0.7, max_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(self, history, context, valid_sources):
        """
        Generate an answer based on conversation history, context, and valid sources.
        This function builds a prompt and calls the OpenAI ChatCompletion API.
        """
        messages = []
        system_prompt = (
            "You are a highly knowledgeable assistant. Answer the user's question based solely on the provided context "
            "and conversation history. Your response should be detailed and include citations where applicable."
        )
        messages.append({"role": "system", "content": system_prompt})
        # Only include context if available.
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        # Append conversation history.
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Call the OpenAI API.
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        answer = response["choices"][0]["message"]["content"]
        return answer

# -------------------- Debugging Helper Function --------------------
def log_embedding_stats(embedding, label="Embedding"):
    norm = np.linalg.norm(embedding)
    logging.debug(f"{label} norm: {norm:.4f}")
    logging.debug(f"{label} sample values: {embedding[:5]}")

# -------------------- Dynamic Chunking and PDF Outline Functions --------------------
def dynamic_chunk_text(text, max_words=700):
    """
    Dynamically splits the input text into chunks based on paragraph boundaries.
    The text is split by double newlines, and paragraphs are combined until reaching max_words.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    current_words = 0
    logging.debug(f"dynamic_chunk_text: Found {len(paragraphs)} paragraphs.")
    for idx, para in enumerate(paragraphs, start=1):
        para_word_count = len(para.split())
        logging.debug(f"Paragraph {idx}: {para_word_count} words.")
        # If a single paragraph is too long, split it further by words.
        if para_word_count >= max_words:
            if current_chunk:
                chunks.append(current_chunk.strip())
                logging.debug(f"Chunk finalized with {current_words} words before long paragraph.")
                current_chunk = ""
                current_words = 0
            words = para.split()
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i+max_words])
                chunks.append(chunk)
                logging.debug(f"Split long paragraph into chunk with {len(chunk.split())} words.")
            continue

        # If adding this paragraph exceeds max_words, finalize the current chunk.
        if current_words + para_word_count > max_words:
            chunks.append(current_chunk.strip())
            logging.debug(f"Chunk finalized with {current_words} words.")
            current_chunk = para + "\n\n"
            current_words = para_word_count
        else:
            current_chunk += para + "\n\n"
            current_words += para_word_count

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        logging.debug(f"Final chunk added with {current_words} words.")

    logging.debug(f"dynamic_chunk_text: Generated {len(chunks)} chunks.")
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
            except Exception as e:
                logging.debug(f"Error getting page number for '{full_title}': {e}")
                page_number = 0
            results.append((full_title, page_number))
            logging.debug(f"Added outline item: '{full_title}' at page {page_number}")
            children = getattr(item, "children", None)
            if children:
                if callable(children):
                    children = children()
                if isinstance(children, list):
                    flatten_outlines(children, pdf_reader, results, full_title)
    return results

def get_outline_sections(pdf_reader):
    if not hasattr(pdf_reader, "outline"):
        logging.debug("PDF does not have an outline attribute.")
        return []
    outline_obj = pdf_reader.outline
    if callable(outline_obj):
        outline_obj = outline_obj()
    if not outline_obj:
        logging.debug("Outline object is empty.")
        return []
    raw_list = flatten_outlines(outline_obj, pdf_reader)
    raw_list.sort(key=lambda x: x[1])
    logging.debug(f"Raw outline list: {raw_list}")
    sections = []
    for i, (title, start_page) in enumerate(raw_list):
        end_page = raw_list[i+1][1] if i < len(raw_list) - 1 else len(pdf_reader.pages)
        sections.append((title, start_page, end_page))
    logging.debug(f"Computed outline sections: {sections}")
    return sections

# -------------------- Persistence Helper Functions --------------------
def load_file_set(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            file_set = set(line.strip() for line in f.readlines())
            logging.debug(f"Loaded {len(file_set)} entries from {filepath}.")
            return file_set
    return set()

def append_to_file(filepath, identifier):
    with open(filepath, "a") as f:
        f.write(identifier + "\n")
    logging.debug(f"Appended identifier {identifier} to {filepath}.")

def remove_from_file(filepath, identifier):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        lines = f.readlines()
    with open(filepath, "w") as f:
        for line in lines:
            if line.strip() != identifier:
                f.write(line)
    logging.debug(f"Removed identifier {identifier} from {filepath}.")

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    logging.debug(f"Computed hash for {filepath}: {file_hash}")
    return file_hash

def load_ingested_files():
    return load_file_set(INGESTED_FILES_PATH)

def save_ingested_files(file_set):
    with open(INGESTED_FILES_PATH, "w") as f:
        for identifier in file_set:
            f.write(identifier + "\n")
    logging.debug(f"Saved {len(file_set)} ingested file identifiers to {INGESTED_FILES_PATH}.")

# -------------------- Conversation Management Functions --------------------
def format_citation(source):
    """
    Given a source string formatted as:
    "Filename.pdf | Section: ... | Pages: ... | Chunk: ..."
    this function extracts the file name, section, and pages (omitting the chunk info)
    and returns a clickable Markdown link.
    """
    parts = [p.strip() for p in source.split("|")]
    if len(parts) < 3:
        # Fallback if not in expected format.
        return f"[{source}](documents/{source})"
    main_file = parts[0]
    section = parts[1]  # Expected to be like "Section: ..."
    pages = parts[2]    # Expected to be like "Pages: ..."
    link_text = f"{main_file} | {section} | {pages}"
    link_href = f"documents/{main_file}"
    return f"[{link_text}]({link_href})"

def create_new_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_thread = {"title": f"Conversation {timestamp}", "history": []}
    st.session_state.setdefault("conversation_threads", []).append(new_thread)
    logging.debug(f"Created new conversation with title: {new_thread['title']}")
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
                short_text = msg["content"].strip()
                thread["title"] = (short_text[:30] + "...") if len(short_text) > 30 else short_text
                logging.debug(f"Renamed conversation to: {thread['title']}")
                return
        thread["title"] = "Untitled conversation"

def handle_delete_conversation():
    delete_choice = st.session_state.get("delete_convo", "")
    if not delete_choice:
        return
    for i, thread in enumerate(st.session_state.get("conversation_threads", [])):
        if thread["title"] == delete_choice:
            del st.session_state["conversation_threads"][i]
            logging.debug(f"Deleted conversation: {delete_choice}")
            if st.session_state.get("active_conversation_index") == i:
                st.session_state["active_conversation_index"] = 0 if st.session_state["conversation_threads"] else None
            elif st.session_state.get("active_conversation_index") is not None and st.session_state["active_conversation_index"] > i:
                st.session_state["active_conversation_index"] -= 1
            break
    try:
        st.experimental_rerun()
    except AttributeError:
        st.write("Please refresh the page manually.")

def get_conversation_html():
    active_thread = get_active_thread()
    if not active_thread:
        return ""
    html = '<div class="chat-container">'
    for msg in active_thread["history"]:
        if msg["role"] == "user":
            html += f'<div class="chat-bubble user-bubble"><strong>User:</strong> {msg["content"]}</div><div class="clearfix"></div>'
        else:
            html += f'<div class="chat-bubble assistant-bubble"><strong>Assistant:</strong> {msg["content"]}</div><div class="clearfix"></div>'
    html += "</div>"
    return html

# -------------------- Document Ingestion Function --------------------
def ingest_documents():
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        st.warning(f"Directory '{docs_dir}' not found.")
        logging.warning(f"Directory '{docs_dir}' not found.")
        return

    ingested_files = load_ingested_files()
    processing_files = load_file_set(PROCESSING_FILES_PATH)
    new_files_found = False
    progress_placeholder = st.empty()

    with st.spinner("Ingesting new documents..."):
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            if os.path.isfile(filepath):
                file_hash = compute_file_hash(filepath)
                if file_hash in ingested_files:
                    logging.debug(f"Skipping already ingested file: {filename}")
                    continue
                if file_hash in processing_files:
                    logging.debug(f"File hash {file_hash} found in processing_files.txt for file {filename}. Removing stale entry and reprocessing.")
                    remove_from_file(PROCESSING_FILES_PATH, file_hash)
                append_to_file(PROCESSING_FILES_PATH, file_hash)
                new_files_found = True

                try:
                    if filename.lower().endswith(".txt"):
                        logging.debug(f"Ingesting text file: {filename}")
                        progress_placeholder.text(f"Ingesting {filename}...")
                        with open(filepath, "r", encoding="utf-8") as f:
                            text = f.read()
                        words = text.split()
                        if len(words) > 300:
                            chunks = dynamic_chunk_text(text, max_words=700)
                            for ch_idx, chunk in enumerate(chunks, start=1):
                                meta_str = f"{filename} | Chunk: {ch_idx}"
                                # Use pre_split=True since our dynamic chunking already splits the text.
                                st.session_state["rag"].ingest_document(chunk, source=meta_str, pre_split=True)
                                logging.debug(f"Ingested chunk {ch_idx} of text file {filename} ({len(chunk.split())} words)")
                                time.sleep(0.3)
                        else:
                            meta_str = f"{filename} | Entire Document"
                            st.session_state["rag"].ingest_document(text, source=meta_str, pre_split=True)
                            logging.debug(f"Ingested entire text file: {filename} ({len(words)} words)")
                        progress_placeholder.text(f"Ingested {filename}")
                        time.sleep(0.5)
                    elif filename.lower().endswith(".pdf"):
                        logging.debug(f"Ingesting PDF file: {filename}")
                        assets_dir = "assets"
                        if not os.path.exists(assets_dir):
                            os.makedirs(assets_dir)
                        asset_pdf_path = os.path.join(assets_dir, filename)
                        if not os.path.exists(asset_pdf_path):
                            shutil.copy(filepath, asset_pdf_path)
                        progress_placeholder.text(f"Ingesting {filename}...")
                        with open(filepath, "rb") as f:
                            pdf_reader = PdfReader(f)
                            sections = get_outline_sections(pdf_reader)
                            if not sections:
                                sections = [("Entire Document", 0, len(pdf_reader.pages))]
                                logging.debug(f"No outline sections found for {filename}, using entire document.")
                            for sec_idx, (section_title, start_page, end_page) in enumerate(sections, start=1):
                                logging.debug(f"Ingesting section {sec_idx} of {filename}: {section_title} (pages {start_page}-{end_page})")
                                progress_placeholder.text(f"Ingesting {filename} [section {sec_idx}: {section_title}]")
                                section_text = ""
                                for pg in range(start_page, end_page):
                                    extracted = pdf_reader.pages[pg].extract_text() or ""
                                    section_text += extracted + "\n\n"
                                chunks = dynamic_chunk_text(section_text, max_words=700)
                                logging.debug(f"Section '{section_title}' split into {len(chunks)} chunks.")
                                for ch_idx, chunk in enumerate(chunks, start=1):
                                    if not chunk.strip():
                                        continue
                                    meta_str = f"{filename} | Section: {section_title} | Pages: {start_page}-{end_page} | Chunk: {ch_idx}"
                                    st.session_state["rag"].ingest_document(chunk, source=meta_str, pre_split=True)
                                    logging.debug(f"Ingested chunk {ch_idx} of section '{section_title}' from {filename} ({len(chunk.split())} words)")
                                    time.sleep(0.3)
                        progress_placeholder.text(f"Ingested {filename}")
                        time.sleep(0.5)
                    else:
                        st.warning(f"Unsupported file type: {filename}")
                        logging.warning(f"Unsupported file type: {filename}")
                    append_to_file(INGESTED_FILES_PATH, file_hash)
                    remove_from_file(PROCESSING_FILES_PATH, file_hash)
                except Exception as e:
                    st.error(f"Error ingesting {filename}: {e}")
                    logging.error(f"Error ingesting {filename}: {e}")
    if new_files_found:
        progress_placeholder.text("")
        st.session_state["rag"].save_index()
        st.success("New documents ingested! Index updated!")
        logging.info("New documents ingested and index updated.")
    else:
        logging.info("No new documents found for ingestion.")

# -------------------- Main Application --------------------
def main():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Aptos+Narrow&display=swap" rel="stylesheet">
        <style>
            html, body {
                font-family: 'Aptos Narrow', sans-serif;
                background-color: #ffffff;
                color: #000000;
            }
            [data-testid="stSidebar"] {
                background-color: #1E4258;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stMarkdown {
                color: #ffffff !important;
            }
            [data-testid="stSidebar"] .stButton button {
                color: #222222 !important;
                background-color: #ffffff !important;
            }
            [data-testid="stSidebar"] .css-1inwz65-control,
            [data-testid="stSidebar"] .css-1inwz65-control * {
                background-color: #ffffff !important;
                color: #222222 !important;
            }
            .chat-container {
                padding: 10px;
                max-width: 800px;
                margin: 20px auto;
            }
            .chat-bubble {
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                display: inline-block;
                max-width: 70%;
                word-wrap: break-word;
            }
            .user-bubble {
                background-color: #265077;
                color: #ffffff;
                float: right;
                clear: both;
            }
            .assistant-bubble {
                background-color: #d3d3d3;
                color: #000000;
                float: left;
                clear: both;
            }
            .clearfix {
                clear: both;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if "rag" not in st.session_state:
        st.session_state["rag"] = RAG(pinecone_index_name=os.environ["PINECONE_INDEX_NAME"])
        logging.debug("Initialized RAG instance.")
    if "generator" not in st.session_state:
        st.session_state["generator"] = Generator()
        logging.debug("Initialized Generator instance.")
    if "conversation_threads" not in st.session_state:
        st.session_state["conversation_threads"] = []
        logging.debug("Initialized conversation threads.")
    if "active_conversation_index" not in st.session_state:
        st.session_state["active_conversation_index"] = None
    if "delete_convo" not in st.session_state:
        st.session_state["delete_convo"] = ""
    if "user_text_input" not in st.session_state:
        st.session_state["user_text_input"] = ""
    if "ingested_files" not in st.session_state:
        st.session_state["ingested_files"] = load_file_set(INGESTED_FILES_PATH)

    if st.session_state.get("active_conversation_index") is None or \
       st.session_state.get("active_conversation_index") >= len(st.session_state["conversation_threads"]):
        new_idx = create_new_conversation()
        st.session_state["active_conversation_index"] = new_idx

    st.title("Workday Compass")
    st.write(
        "This app retrieves information from customer-facing and validated Workday content, including administrator guides, "
        "knowledge articles, release notes, contributed solutions, and posted questions with answers, among others. "
        "If no relevant document is found, the response is based on general knowledge. Type your message below and click Send. "
        "Citations (if available) will be appended to the answer."
    )

    st.sidebar.title("Conversations")
    convo_titles = [t["title"] for t in st.session_state["conversation_threads"]]
    active_idx = st.session_state["active_conversation_index"]
    if convo_titles:
        current_selection = st.sidebar.selectbox(
            "Select a conversation to view:",
            options=convo_titles,
            index=active_idx,
        )
        chosen_idx = convo_titles.index(current_selection)
        if chosen_idx != active_idx:
            st.session_state["active_conversation_index"] = chosen_idx

        delete_choice = st.sidebar.selectbox("Select a conversation to delete:", options=convo_titles)
        st.session_state["delete_convo"] = delete_choice
        if st.sidebar.button("Delete Conversation"):
            handle_delete_conversation()
    else:
        st.sidebar.write("No conversations yet.")

    if st.sidebar.button("Start New Conversation"):
        new_idx = create_new_conversation()
        st.session_state["active_conversation_index"] = new_idx

    ingest_documents()

    st.markdown("### Conversation")
    conversation_placeholder = st.empty()
    conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Enter your message",
            key="user_text_input",
            placeholder="Type your message here...",
            height=100,
            label_visibility="collapsed",
        )
        send_clicked = st.form_submit_button("Send")

    if send_clicked and user_input.strip():
        active_thread = get_active_thread()
        if active_thread:
            msg = user_input.strip()
            active_thread["history"].append({"role": "user", "type": "text", "content": msg})
            rename_conversation_auto(active_thread)
            conversation_html = get_conversation_html()
            conversation_placeholder.markdown(conversation_html, unsafe_allow_html=True)

            logging.debug(f"User query: {msg}")

            temp_msg = "<p style='font-style: italic; color: #555555; margin: 0;'>Searching the knowledge base...</p>"
            conversation_placeholder.markdown(conversation_html + temp_msg, unsafe_allow_html=True)

            start_time = time.time()
            results = st.session_state["rag"].query(msg, top_k=3)
            elapsed = time.time() - start_time
            logging.debug(f"Query returned {len(results)} results in {elapsed:.2f} seconds.")
            if results:
                top_chunk = results[0]
                logging.debug(f"Top chunk content: {top_chunk.get('text', '')}")
            if elapsed < 10:
                time.sleep(10 - elapsed)

            all_scores = [r.get("score", 0) for r in results]
            logging.debug("All result scores: " + ", ".join([f"{score:.4f}" for score in all_scores]))
            best_score = max(all_scores) if all_scores else 0
            logging.debug(f"Best similarity score: {best_score:.4f}")
            best_distance = 1 - best_score
            unrelated_distance_threshold = 1.0
            logging.debug(f"Best distance: {best_distance:.4f} (threshold: {unrelated_distance_threshold})")
            if best_distance > unrelated_distance_threshold:
                logging.debug("Best distance exceeds threshold, treating as no relevant documents found.")
                filtered_results = []
            else:
                filtered_results = [r for r in results if r.get("score", 0) >= SIMILARITY_THRESHOLD]
            logging.debug(f"Filtered to {len(filtered_results)} results using similarity threshold {SIMILARITY_THRESHOLD}.")

            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

            if not filtered_results:
                context = ""
                valid_sources = []
                logging.debug("No relevant documents found based on the similarity threshold and best distance.")
            else:
                retrieved_chunks = [r.get("text", "") for r in filtered_results]
                context = " ".join(retrieved_chunks)
                sources = [r.get("source", "unknown") for r in filtered_results]
                valid_sources = [s for s in sources if s and s.strip().lower() != "unknown"]
                logging.debug(f"Relevant documents found with sources: {valid_sources}")

            full_answer = st.session_state["generator"].generate_answer(active_thread["history"], context, valid_sources)
            if not filtered_results:
                full_answer = (
                    "We could not find relevant information from our internal knowledge base regarding your query. "
                    "Based on general knowledge, here is what we found:\n\n"
                ) + full_answer
            elif valid_sources:
                clickable_citations = ", ".join(format_citation(s) for s in valid_sources)
                full_answer += "\n\n**Citations:** " + clickable_citations

            assistant_placeholder = st.empty()
            partial_text = ""
            for char in full_answer:
                partial_text += char
                temp_html = get_conversation_html() + f'<div class="chat-bubble assistant-bubble"><strong>Assistant:</strong> {partial_text}</div><div class="clearfix"></div>'
                conversation_placeholder.markdown(temp_html, unsafe_allow_html=True)
                time.sleep(0.005)
            active_thread["history"].append({"role": "assistant", "type": "text", "content": full_answer})
            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)
            logging.debug("Assistant response generated and appended to conversation history.")

if __name__ == "__main__":
    main()
