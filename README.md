# Document Intelligence & Mortgage RAG Pipeline

A privacy-focused, open-source Retrieval-Augmented Generation (RAG) assistant designed for extracting and querying complex financial documents, loan worksheets, and legal contracts.

Built with **LlamaIndex**, **PyMuPDF**, **BGE Embeddings**, **Mistral-7B (GGUF)**, and **Gradio**.

---

## Overview

Financial packets (such as loan estimates, closing disclosures, and service contracts) are dense, multi-page documents where generic LLMs frequently hallucinate exact figures or lose context across page boundaries. 

This project implements an end-to-end local RAG pipeline that:
- Ingests multi-page PDFs with page-level metadata tracking.
- Indexes document chunks using open-source dense vector embeddings.
- Executes local inference with a quantized 7B language model without sending sensitive client data to external cloud APIs.
- Serves an interactive Gradio interface featuring confidence scoring and verified page-level source citations.

---

## System Architecture

```text
[PDF Document]
      │
      ▼
[PyMuPDF (fitz)] ──────► Page-level text extraction + metadata tagging
      │                  (filename, page_number, doc_type)
      ▼
[BGE Embeddings] ──────► 384-dimensional dense vectors (bge-small-en-v1.5)
      │
      ▼
[VectorStoreIndex] ────► In-memory vector store & top-k semantic search
      │
      ▼
[Context Builder] ─────► Compact synthesis with citation formatting
      │
      ▼
[Mistral-7B (GGUF)] ───► Local GPU/CPU inference via llama.cpp (0 API calls)
      │
      ▼
[Gradio Web UI] ───────► Answer + Confidence % + Source Snippets
```

---

## Key Features

- **Zero Cloud API Dependencies:** Runs entirely on open-source weights. LLM inference is handled locally by `Mistral-7B-Instruct-v0.2` (quantized in `Q4_K_M` GGUF format via `llama.cpp`), ensuring full data privacy for confidential financial documents.
- **Page-Level Source Grounding:** Every generated response displays exact page numbers, file sources, and excerpt snippets to enable human verification.
- **Dynamic Confidence Scoring:** Computes retrieval confidence percentages derived from semantic similarity scores before generating responses.
- **Robust Ingestion:** Uses PyMuPDF for fast, memory-efficient PDF text extraction, skipping blank pages and attaching clean structural metadata.

---

## Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Orchestration** | LlamaIndex | Industry-standard framework for document retrieval and indexing pipelines |
| **PDF Extraction** | PyMuPDF (`fitz`) | Fast, lightweight C-based PDF text parser |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | Top-performing lightweight embedding model on MTEB benchmarks |
| **Language Model** | `Mistral-7B-Instruct-v0.2` | High-performing instruction-tuned model, quantized to 4 bits (~4.3 GB) |
| **Inference Engine** | `llama.cpp` (`llama-cpp-python`) | Hardware-optimized C++ inference engine with CUDA/Metal acceleration |
| **User Interface** | Gradio | Clean, responsive web interface with live document upload and inspection |

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA 12.x or Apple Silicon (Metal) for accelerated inference.

### 1. Clone the Repository
```bash
git clone https://github.com/Ayjeren004/document-intelligence-rag.git
cd document-intelligence-rag
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for CUDA GPU acceleration:** If running with NVIDIA CUDA, install `llama-cpp-python` with CUDA support:
> ```bash
> pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
> ```

---

## Usage

Run the Gradio web application:
```bash
python app.py
```

1. Open the local URL (`http://127.0.0.1:7860`) or public share URL in your browser.
2. Upload a PDF document (e.g. loan worksheet, contract, or financial report).
3. Click **Index Document**.
4. Ask domain-specific questions (e.g., *"What is the total estimated monthly payment?"* or *"What are the late payment penalties?"*).
5. Inspect the answer, match confidence, and cited page sources in the sidebar.

---

## Project Structure

```text
├── app.py                 # Main application script (Ingestion, RAG engine, Gradio UI)
├── requirements.txt       # Python dependencies
├── .gitignore             # Standard gitignore rules
└── README.md              # Project documentation
```

---

## License
MIT License.
