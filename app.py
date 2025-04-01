import os
import time
import streamlit as st
from datetime import datetime
from PyPDF2 import PdfReader
import numpy as np
import asyncio
import shutil
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file.
load_dotenv()

# -------------------------------
# Pinecone Initialization (New API)
# -------------------------------
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-west-2")

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # Adjust to your embedding model (e.g., 384 for all-MiniLM-L6-v2)
        metric='cosine',  # Using cosine similarity
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

# -------------------------------
# Other Imports and Configurations
# -------------------------------
from rag import RAG  # Ensure your rag.py is updated to use Pinecone logic and returns a score with each result.
from generator import Generator

# Disable Streamlit's file watcher to help avoid Torch errors.
os.environ["ST_STREAMLIT_WATCHED_FILES"] = ""

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Set thresholds.
UNRELATED_DISTANCE_THRESHOLD = 1.0
FOLLOWUP_SIMILARITY_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.7  # Only use results with a score >= 0.7

# File persistence paths for tracking ingestion.
INGESTED_FILES_PATH = "ingested_files.txt"
PROCESSING_FILES_PATH = "processing_files.txt"

# ---------- Helper Functions for File Ingestion Persistence ----------
def load_file_set(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return set(line.strip() for line in f.readlines())
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

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_ingested_files():
    return load_file_set(INGESTED_FILES_PATH)

def save_ingested_files(file_set):
    with open(INGESTED_FILES_PATH, "w") as f:
        for identifier in file_set:
            f.write(identifier + "\n")

# ---------- Helper Function for Citation Formatting ----------
def format_citation(source):
    if " (page " in source:
        file_name, page_info = source.split(" (page ", 1)
        page_info = " (page " + page_info
    else:
        file_name = source
        page_info = ""
    if file_name.lower().endswith(".pdf"):
        link = f"/assets/{file_name}"
    else:
        link = f"documents/{file_name}"
    return f"[{file_name}]({link}){page_info}"

# ---------- Conversation Management Functions ----------
def create_new_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"Conversation {timestamp}"
    new_thread = {"title": title, "history": []}
    st.session_state["conversation_threads"].append(new_thread)
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
                thread["title"] = (text[:30] + "...") if len(text) > 30 else text
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
            elif st.session_state.get("active_conversation_index") is not None and st.session_state["active_conversation_index"] > i:
                st.session_state["active_conversation_index"] -= 1
            break
    try:
        st.experimental_rerun()
    except AttributeError:
        st.write("Please refresh the page manually.")

def get_conversation_html():
    active_thread = get_active_thread()
    if active_thread is None:
        return ""
    html = '<div class="chat-container">'
    for msg in active_thread["history"]:
        if msg["role"] == "user":
            html += f'<div class="chat-bubble user-bubble"><strong>User:</strong> {msg["content"]}</div><div class="clearfix"></div>'
        else:
            html += f'<div class="chat-bubble assistant-bubble"><strong>Assistant:</strong> {msg["content"]}</div><div class="clearfix"></div>'
    html += '</div>'
    return html

def is_followup_query(history):
    user_msgs = [m for m in history if m["role"] == "user"]
    if len(user_msgs) < 2:
        return False
    previous_query = user_msgs[-2]["content"]
    new_query = user_msgs[-1]["content"]
    new_emb = st.session_state["rag"].model.encode([new_query])[0]
    prev_emb = st.session_state["rag"].model.encode([previous_query])[0]
    sim = np.dot(new_emb, prev_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(prev_emb))
    return sim >= FOLLOWUP_SIMILARITY_THRESHOLD

# ---------- Document Ingestion Function ----------
def ingest_documents():
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        st.warning(f"Directory '{docs_dir}' not found. Please create it and add your documents.")
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
                if file_hash in ingested_files or file_hash in processing_files:
                    continue  # Skip already processed files.
                append_to_file(PROCESSING_FILES_PATH, file_hash)
                new_files_found = True
                try:
                    if filename.lower().endswith(".txt"):
                        progress_placeholder.text(f"Ingesting {filename}...")
                        with open(filepath, "r", encoding="utf-8") as f:
                            text = f.read()
                        st.session_state["rag"].ingest_document(text, source=filename)
                        progress_placeholder.text(f"Ingested {filename}")
                        time.sleep(0.5)
                    elif filename.lower().endswith(".pdf"):
                        assets_dir = "assets"
                        if not os.path.exists(assets_dir):
                            os.makedirs(assets_dir)
                        asset_pdf_path = os.path.join(assets_dir, filename)
                        if not os.path.exists(asset_pdf_path):
                            shutil.copy(filepath, asset_pdf_path)
                        progress_placeholder.text(f"Ingesting {filename}...")
                        with open(filepath, "rb") as f:
                            pdf_reader = PdfReader(f)
                            for i, page in enumerate(pdf_reader.pages, start=1):
                                progress_placeholder.text(f"Ingesting {filename} [page {i}]")
                                page_text = page.extract_text()
                                if page_text:
                                    st.session_state["rag"].ingest_document(page_text, source=f"{filename} (page {i})")
                                time.sleep(0.3)
                        progress_placeholder.text(f"Ingested {filename}")
                        time.sleep(0.5)
                    else:
                        st.warning(f"Unsupported file type: {filename}")
                    append_to_file(INGESTED_FILES_PATH, file_hash)
                    remove_from_file(PROCESSING_FILES_PATH, file_hash)
                except Exception as e:
                    st.error(f"Error ingesting {filename}: {e}")
    if new_files_found:
        progress_placeholder.text("")
        st.session_state["rag"].save_index()
        st.success("New documents ingested! Index updated!")
    else:
        pass

# ---------- Main Function ----------
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
            /* Sidebar styling */
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
            /* Chat container styling without background color */
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
        st.session_state["rag"] = RAG(pinecone_index_name=PINECONE_INDEX_NAME)
    if "generator" not in st.session_state:
        st.session_state["generator"] = Generator()
    if "conversation_threads" not in st.session_state:
        st.session_state["conversation_threads"] = []
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
            label_visibility="collapsed"
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

            # Insert temporary plain text immediately below the user's query.
            temp_msg = "<p style='font-style: italic; color: #555555; margin: 0;'>Searching the knowledge base...</p>"
            # Append the temporary text right after the conversation HTML.
            conversation_placeholder.markdown(conversation_html + temp_msg, unsafe_allow_html=True)

            # Record start time and enforce a minimum display duration.
            start_time = time.time()
            results = st.session_state["rag"].query(msg, top_k=3)
            elapsed = time.time() - start_time
            if elapsed < 2:
                time.sleep(2 - elapsed)

            # Filter results based on similarity threshold.
            filtered_results = [r for r in results if r.get("score", 0) >= SIMILARITY_THRESHOLD]

            # Re-render conversation without the temporary text.
            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

            if not filtered_results:
                context = (
                    "We could not find relevant information from our internal knowledge base regarding your query. "
                    "Based on general knowledge, here is what we found:"
                )
                valid_sources = []
            else:
                retrieved_chunks = [r.get("text", "") for r in filtered_results]
                context = " ".join(retrieved_chunks)
                sources = [r.get("source", "unknown") for r in filtered_results]
                valid_sources = [s for s in sources if s and s.strip().lower() != "unknown"]

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

if __name__ == "__main__":
    main()
