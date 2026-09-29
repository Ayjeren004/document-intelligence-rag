"""
Capstone Project: Full RAG UI for Mortgage & Financial Documents
Stack: LlamaIndex, PyMuPDF, BGE Embeddings, Mistral-7B / Gemini, Gradio UI
"""

import os
import fitz  # PyMuPDF
import gradio as gr
from typing import List

# LlamaIndex core imports
from llama_index.core import Document, VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# -------------------------------------------------------------
# 1. CONFIGURE EMBEDDINGS & LLM
# -------------------------------------------------------------
print("Loading open-source BGE embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GEMINI_API_KEY":
    from llama_index.llms.gemini import Gemini
    print("Using Gemini LLM backend...")
    llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    Settings.llm = llm
else:
    from llama_index.llms.llama_cpp import LlamaCPP
    model_path = "/content/mistral.gguf"
    print("Using local Mistral-7B GGUF backend...")
    try:
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=300,
            context_window=2048,
            model_kwargs={"n_gpu_layers": 33 if os.path.exists("/usr/local/cuda") else 0}
        )
        Settings.llm = llm
    except Exception as e:
        print(f"Notice on local LLM: {e}")

# Global storage for document index
index_store = None
current_doc_name = "None"


# -------------------------------------------------------------
# 2. DOCUMENT INGESTION & METADATA EXTRACTION
# -------------------------------------------------------------
def ingest_document(file):
    """Parses PDF page-by-page, tags metadata, and builds Vector Index."""
    global index_store, current_doc_name
    if file is None:
        return "⚠️ No file uploaded. Please select a PDF."

    try:
        doc = fitz.open(file.name)
        documents = []
        filename = os.path.basename(file.name)
        current_doc_name = filename

        for i, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue  # Skip blank pages

            # Attach structured page metadata for filtering and citation
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "filename": filename,
                        "page_number": i + 1,
                        "total_pages": len(doc),
                        "doc_type": "Mortgage/Financial Document"
                    }
                )
            )
        doc.close()

        if not documents:
            return f"❌ Could not extract text from {filename}. If it is a scanned image, please ensure OCR is applied."

        # Build Vector Store Index
        index_store = VectorStoreIndex.from_documents(documents)
        return f"✅ '{filename}' successfully indexed! ({len(documents)} pages processed with metadata)."
    except Exception as e:
        return f"❌ Error indexing document: {e}"


# -------------------------------------------------------------
# 3. RETRIEVAL & CITATION ENGINE
# -------------------------------------------------------------
def rag_chat(user_question, chat_history):
    """Executes RAG pipeline, returning answer, cited sources, and confidence score."""
    global index_store
    if index_store is None:
        return "", chat_history, "⚠️ Please upload and index a PDF first.", "N/A"

    if not user_question.strip():
        return "", chat_history, "", "N/A"

    try:
        # Retrieve top 2 most relevant chunks
        retriever = VectorIndexRetriever(index=index_store, similarity_top_k=2)
        retrieved_nodes = retriever.retrieve(user_question)

        if not retrieved_nodes:
            answer = "I could not find relevant information in the uploaded document to answer that question."
            chat_history.append((user_question, answer))
            return "", chat_history, "No matching document sections found.", "0%"

        # Format sources & estimate confidence from cosine similarity
        sources_text = ""
        avg_score = 0.0
        for i, node in enumerate(retrieved_nodes):
            score = node.score if hasattr(node, "score") and node.score else 0.85
            avg_score += score
            page_num = node.metadata.get("page_number", "Unknown")
            fname = node.metadata.get("filename", "Document")
            snippet = node.get_text()[:180].replace("\n", " ").strip()
            sources_text += f"📌 **Source {i+1}:** {fname} (Page {page_num}) — *Match Score: {score:.2f}*\n"
            sources_text += f"> *\"{snippet}...\"*\n\n"

        confidence_pct = f"{min(100, int((avg_score / len(retrieved_nodes)) * 100))}%"

        # Synthesize grounded answer
        synthesizer = get_response_synthesizer(response_mode="compact")
        response = synthesizer.synthesize(user_question, nodes=retrieved_nodes)
        answer = str(response).strip()

        # Update chat history
        chat_history.append((user_question, answer))
        return "", chat_history, sources_text, confidence_pct

    except Exception as e:
        return "", chat_history, f"Error: {e}", "Error"


# -------------------------------------------------------------
# 4. PRODUCT-READY GRADIO UI
# -------------------------------------------------------------
def create_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("# 🏦 Mortgage & Financial Document RAG Assistant")
        gr.Markdown("Upload complex loan worksheets, contracts, or tax returns. Ask questions and get grounded answers with page-level citations.")

        with gr.Row():
            # Left Panel: File Ingestion & Metadata Inspection
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Document (PDF)", file_types=[".pdf"])
                ingest_btn = gr.Button("⚡ Index Document", variant="primary")
                status_box = gr.Textbox(label="Ingestion Status", interactive=False)

                gr.Markdown("### 🔍 Retrieval Inspection")
                confidence_box = gr.Textbox(label="Retrieval Confidence", interactive=False)
                sources_box = gr.Markdown(label="Cited Sources")

            # Right Panel: Chat Interface
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Interactive Conversation", height=450)
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask a question (e.g. 'What is the total estimated monthly payment?')",
                        label="User Question",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat History")

        # Event triggers
        ingest_btn.click(ingest_document, inputs=[file_input], outputs=[status_box])
        submit_btn.click(
            rag_chat,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot, sources_box, confidence_box]
        )
        query_input.submit(
            rag_chat,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot, sources_box, confidence_box]
        )
        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
