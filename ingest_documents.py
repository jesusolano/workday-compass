import os
from rag import RAG
from PyPDF2 import PdfReader

# Create an instance of RAG
rag = RAG()

docs_dir = "documents"  # Directory where your documents reside

if os.path.exists(docs_dir):
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if os.path.isfile(filepath):
            if filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"Ingesting {filename}...")
                rag.ingest_document(text)
            elif filename.lower().endswith(".pdf"):
                with open(filepath, "rb") as f:
                    pdf_reader = PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                print(f"Ingesting {filename}...")
                rag.ingest_document(text)
            else:
                print(f"Skipping unsupported file: {filename}")
else:
    print(f"Directory {docs_dir} not found.")

# Save the index and chunks to disk
rag.save_index()
print("Document ingestion complete. Index and chunks saved.")
