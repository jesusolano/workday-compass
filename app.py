import os
import time
import streamlit as st
from rag import RAG
from generator import Generator
from datetime import datetime
from PyPDF2 import PdfReader
import numpy as np

# Set thresholds.
UNRELATED_DISTANCE_THRESHOLD = 1.0
FOLLOWUP_SIMILARITY_THRESHOLD = 0.7

# Persistent file paths for index, chunks, and ingested files.
INDEX_PATH = "rag_index.pkl"
CHUNKS_PATH = "rag_chunks.pkl"
INGESTED_FILES_PATH = "ingested_files.txt"

# --- Helper Functions for Persistent Ingestion Tracking ---
def load_ingested_files():
    if os.path.exists(INGESTED_FILES_PATH):
        with open(INGESTED_FILES_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []

def save_ingested_files(file_list):
    with open(INGESTED_FILES_PATH, "w") as f:
        for filename in file_list:
            f.write(filename + "\n")

# --- Other Helper Functions ---
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
    st.experimental_rerun()

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
    """Return True if the new user query is a follow-up (cosine similarity >= threshold) to the previous user message."""
    user_msgs = [m for m in history if m["role"] == "user"]
    if len(user_msgs) < 2:
        return False
    previous_query = user_msgs[-2]["content"]
    new_query = user_msgs[-1]["content"]
    new_emb = st.session_state["rag"].model.encode([new_query])[0]
    prev_emb = st.session_state["rag"].model.encode([previous_query])[0]
    sim = np.dot(new_emb, prev_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(prev_emb))
    return sim >= FOLLOWUP_SIMILARITY_THRESHOLD

# --- Function to Ingest Documents ---
def ingest_documents():
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        st.warning(f"Directory '{docs_dir}' not found. Please create it and add your documents.")
        return

    st.write(f"Checking for new documents in the **{docs_dir}** directory...")
    new_files_found = False
    with st.spinner("Ingesting new documents..."):
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            if os.path.isfile(filepath) and filename not in st.session_state.get("ingested_files", []):
                new_files_found = True
                try:
                    if filename.lower().endswith(".txt"):
                        with open(filepath, "r", encoding="utf-8") as f:
                            text = f.read()
                        source = filename if filename.strip() else "unknown"
                        st.session_state["rag"].ingest_document(text, source=source)
                        st.info(f"Ingested: {filename}")
                    elif filename.lower().endswith(".pdf"):
                        with open(filepath, "rb") as f:
                            pdf_reader = PdfReader(f)
                            for i, page in enumerate(pdf_reader.pages, start=1):
                                page_text = page.extract_text()
                                if page_text:
                                    source = f"{filename} (page {i})"
                                    st.session_state["rag"].ingest_document(page_text, source=source)
                                    st.info(f"Ingested: {filename} (page {i})")
                    else:
                        st.warning(f"Unsupported file type: {filename}")
                    st.session_state["ingested_files"].append(filename)
                except Exception as e:
                    st.error(f"Error ingesting {filename}: {e}")
    if new_files_found:
        st.session_state["rag"].save_index()
        save_ingested_files(st.session_state["ingested_files"])
        st.success("New documents ingested and index saved!")
    else:
        st.write("No new documents found. Using previously ingested documents.")

# --- Main Function ---
def main():
    # Session State Initialization
    if "rag" not in st.session_state:
        st.session_state["rag"] = RAG(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH)
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
        st.session_state["ingested_files"] = load_ingested_files()

    # Global CSS: Import Aptos font and apply globally.
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Aptos&display=swap');
            html, body, [class*="css"]  {
                font-family: 'Aptos', sans-serif;
            }
        </style>
        """, unsafe_allow_html=True)

    # Ensure at least one conversation exists.
    if st.session_state.get("active_conversation_index") is None or \
       st.session_state.get("active_conversation_index") >= len(st.session_state.get("conversation_threads", [])):
        new_idx = create_new_conversation()
        st.session_state["active_conversation_index"] = new_idx

    # App Title & Description
    st.title("Conversational RAG Demo App")
    st.write(
        "Welcome! This app retrieves information from your ingested documents. "
        "If no relevant document is found, the response is based on general knowledge. "
        "Type your message below and click Send. The assistant's response will appear with a simulated typing effect. "
        "Citations (if available) will be appended to the answer."
    )

    # Sidebar: Conversation Management
    st.sidebar.title("Conversations")
    convo_titles = [t["title"] for t in st.session_state.get("conversation_threads", [])]
    active_idx = st.session_state.get("active_conversation_index", 0)
    if convo_titles:
        current_selection = st.sidebar.selectbox(
            "Select a conversation to view:",
            options=convo_titles,
            index=active_idx
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

    # Trigger document ingestion.
    ingest_documents()

    # Custom CSS for Chat Bubbles & Input Layout
    st.markdown(
        """
        <style>
        .chat-container { max-width: 800px; margin: 20px auto; padding: 10px; }
        .chat-bubble { padding: 10px; border-radius: 10px; margin-bottom: 10px; display: inline-block; max-width: 70%; word-wrap: break-word; }
        .user-bubble { background-color: #dcf8c6; float: right; clear: both; }
        .assistant-bubble { background-color: #ececec; float: left; clear: both; }
        .clearfix { clear: both; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the Conversation
    st.markdown("### Conversation")
    conversation_placeholder = st.empty()
    conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

    # Input Form: Text Area and Send Button
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
            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

            # Retrieve document context.
            query_embedding = st.session_state["rag"].model.encode([msg]).astype("float32")
            distances, indices = st.session_state["rag"].index.search(query_embedding, 3)
            best_distance = distances[0][0] if distances.size > 0 else None

            # Determine if the new query is a follow-up.
            history = active_thread["history"]
            if len([m for m in history if m["role"] == "user"]) > 1:
                prev_query = history[-2]["content"]
                new_query = history[-1]["content"]
                prev_emb = st.session_state["rag"].model.encode([prev_query])[0]
                new_emb = st.session_state["rag"].model.encode([new_query])[0]
                sim = np.dot(new_emb, prev_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(prev_emb))
                is_followup = sim >= FOLLOWUP_SIMILARITY_THRESHOLD
            else:
                is_followup = False

            if best_distance is not None and best_distance > UNRELATED_DISTANCE_THRESHOLD and not is_followup:
                context = ("We could not find relevant information from our internal knowledge base regarding your query. "
                           "Based on general knowledge, here is what we found:")
                sources = []
            else:
                retrieved_chunks = [st.session_state["rag"].chunks[idx]["text"] for idx in indices[0] if idx < len(st.session_state["rag"].chunks)]
                context = " ".join(retrieved_chunks)
                sources = [st.session_state["rag"].chunks[idx]["source"] for idx in indices[0] if idx < len(st.session_state["rag"].chunks)]
                valid_sources = [s for s in sources if s and s.strip().lower() != "unknown"]
                if valid_sources:
                    context += "\n\n**Citations:** " + ", ".join(valid_sources)
                sources = valid_sources

            # Generate the assistant's answer using the conversation history.
            full_answer = st.session_state["generator"].generate_answer(active_thread["history"], context,
                                                                          sources if best_distance is not None and best_distance <= UNRELATED_DISTANCE_THRESHOLD else [])
            if best_distance is not None and best_distance > UNRELATED_DISTANCE_THRESHOLD and not is_followup:
                full_answer = ("We could not find relevant information from our internal knowledge base regarding your query. "
                              "Based on general knowledge, here is what we found:\n\n") + full_answer
            elif not is_followup and sources:
                full_answer += "\n\n**Citations:** " + ", ".join(sources)

            # Simulate typing effect.
            assistant_placeholder = st.empty()
            partial_text = ""
            for char in full_answer:
                partial_text += char
                temp_html = get_conversation_html() + f'<div class="chat-bubble assistant-bubble"><strong>Assistant:</strong> {partial_text}</div><div class="clearfix"></div>'
                conversation_placeholder.markdown(temp_html, unsafe_allow_html=True)
                time.sleep(0.005)
            active_thread["history"].append({"role": "assistant", "type": "text", "content": full_answer})
            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)
        # Conversation history persists without forcing a rerun.

if __name__ == "__main__":
    main()
