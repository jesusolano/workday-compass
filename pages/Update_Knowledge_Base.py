# Update Knowledge Base

import os
import time
import streamlit as st
from ingestion_utils import ingest_documents, RAG

def main():
    st.title("Document Upload")
    st.write("Upload your documents to update the internal knowledge base.")

    # Ensure 'rag' is initialized in session state using Streamlit secrets
    if "rag" not in st.session_state:
        st.session_state["rag"] = RAG(pinecone_index_name=st.secrets["PINECONE_INDEX_NAME"])

    # File uploader that accepts multiple files
    uploaded_files = st.file_uploader("Choose documents", accept_multiple_files=True)

    if uploaded_files:
        docs_dir = "documents"
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name}")

        # Trigger the ingestion process after uploads
        with st.spinner("Ingesting documents..."):
            ingest_documents()  # Process files in the "documents" folder
            time.sleep(1)  # Optional pause for processing
        st.success("Documents ingested and index updated!")

if __name__ == "__main__":
    main()
