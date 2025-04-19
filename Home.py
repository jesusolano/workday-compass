# Home

import os
import time
import streamlit as st
import logging
import sys
from ingestion_utils import (
    RAG,
    Generator,
    get_conversation_html,
    create_new_conversation,
    get_active_thread,
    handle_delete_conversation,
    rename_conversation_auto,
    format_citation
)

def main():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Aptos+Narrow&display=swap" rel="stylesheet">
        <style>
            /* Styling for the conversation page */
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

    # Initialize session state for RAG and Generator if not already set
    if "rag" not in st.session_state:
        st.session_state["rag"] = RAG(pinecone_index_name=st.secrets["PINECONE_INDEX_NAME"])
    if "generator" not in st.session_state:
        st.session_state["generator"] = Generator()

    # Initialize conversation state if needed
    if "conversation_threads" not in st.session_state:
        st.session_state["conversation_threads"] = []
    if "active_conversation_index" not in st.session_state:
        st.session_state["active_conversation_index"] = None
    if "delete_convo" not in st.session_state:
        st.session_state["delete_convo"] = ""
    if "user_text_input" not in st.session_state:
        st.session_state["user_text_input"] = ""

    # Ensure an active conversation exists
    if (st.session_state.get("active_conversation_index") is None or
       st.session_state["active_conversation_index"] >= len(st.session_state["conversation_threads"])):
        new_idx = create_new_conversation()
        st.session_state["active_conversation_index"] = new_idx

    st.title("Workday Compass")
    st.write(
        "This app retrieves information from validated Workday content. "
        "If no relevant document is found, the response is based on general knowledge."
    )

    # Sidebar for conversation selection
    # ─── Retrieval Controls ──────────────────────────────────────
    st.sidebar.markdown("## Retrieval Settings")
    alpha = st.sidebar.slider(
        "Dense vs Sparse α",
        0.0, 1.0, 0.5, step=0.05,
        help=(
            "Choose how much to trust semantic matches vs. exact keywords:\n\n"
            "• Move right (toward 1.0) to lean on meaning-based (dense) search.\n\n"
            "• Move left (toward 0.0) to lean on exact-word (sparse) search."
        )
    )
    dense_k = st.sidebar.slider(
        "Dense top_k",
        1, 10, 5,
        help=(
            "How many top results to pull from the semantic (dense) index.\n"
            "Higher values may find more related info but take slightly longer."
        )
    )
    sparse_k = st.sidebar.slider(
        "Sparse top_k",
        1, 10, 5,
        help=(
            "How many top results to pull from the keyword (sparse) index.\n"
            "Higher values capture more exact matches but may slow things down."
        )
    )
    final_k   = st.sidebar.slider(
        "Results to Display",
        1, 5, 3,
        help="How many top results (after merging) to show in the chat."
    )
    threshold = st.sidebar.slider(
        "Relevance Threshold",
        0.0, 1.0, 0.50, step=0.01,
        help="Minimum combined score for a result to be considered “relevant.”"
    )

    # ─── Sidebar: Conversations ─────────────────────────────────
    st.sidebar.title("Conversations")
    convo_titles = [t["title"] for t in st.session_state["conversation_threads"]]
    active_idx = st.session_state["active_conversation_index"]
    if convo_titles:
        current_selection = st.sidebar.selectbox("Select a conversation to view:", options=convo_titles, index=active_idx)
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

    # ---------------- Conversation Container ----------------
    chat_container = st.container()
    with chat_container:
        st.markdown("### Conversation")
        conversation_placeholder = st.empty()
        conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

    # ---------------- Chat Input Form ----------------
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
            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

            # Show a temporary "searching" message
            temp_msg = "<p style='font-style: italic; color: #555555; margin: 0;'>Searching the knowledge base...</p>"
            conversation_placeholder.markdown(get_conversation_html() + temp_msg, unsafe_allow_html=True)

            # Query Pinecone for top results
            start_time = time.time()
            #results = st.session_state["rag"].query(msg, top_k=3)
            # 1) Hybrid retrieval
            raw_results = st.session_state["rag"].query(
                query_text=msg,
                top_k=final_k,      # pull exactly final_k if you want to rely purely on top_k
                alpha=alpha,
                dense_k=dense_k,
                sparse_k=sparse_k
            )
            # 2) Apply threshold & limit to final_k
            filtered_results = [
                r for r in raw_results
                if r["score"] >= threshold
            ][:final_k]

            elapsed = time.time() - start_time
            if elapsed < 5:
                time.sleep(5 - elapsed)

            #sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            #filtered_results = [r for r in sorted_results if r.get("score", 0) >= 0.55]
            # (filtered_results now honors both your threshold slider and how many you display)

            context = " ".join(r["text"] for r in filtered_results) if filtered_results else ""
            answer = st.session_state["generator"].generate_answer(active_thread["history"], context, [])

            if not filtered_results:
                answer_intro = (
                    "We could not find relevant information from our internal knowledge base regarding your query. "
                    "Based on general knowledge, here is what we found:\n\n"
                )
            else:
                answer_intro = ""
            answer = answer_intro + answer

            # "Type" out the answer gradually
            partial_text = ""
            for char in answer:
                partial_text += char
                typed_html = (
                    get_conversation_html() +
                    f'<div class="chat-bubble assistant-bubble"><strong>Assistant:</strong> {partial_text}</div>'
                    f'<div class="clearfix"></div>'
                )
                conversation_placeholder.markdown(typed_html, unsafe_allow_html=True)
                time.sleep(0.0025)

            # Prepare one citation dict for EACH filtered result
            citations = []
            for res in filtered_results:
                citations.append({
                    "header": format_citation(res["source"]),
                    "content": res["text"]
                })

            # Append the assistant response (with citation if available) to the conversation history
            assistant_message = {"role": "assistant", "type": "text", "content": answer}
            if citations:
                assistant_message["citations"] = citations
            active_thread["history"].append(assistant_message)
            conversation_placeholder.markdown(get_conversation_html(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
