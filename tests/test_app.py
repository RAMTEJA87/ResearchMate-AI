"""
=============================================================================
ResearchMate AI — Unit & Integration Test Suite (tests/test_app.py)
=============================================================================
Tests:
- Flask homepage (GET /)
- Missing question handling (POST /chat)
- Missing Groq API key handling
- Invalid file extension rejection
- Valid PDF ingestion & chunk metadata preservation
- Empty PDF / OCR detection
- 0-indexed to 1-indexed display page numbering conversion
- FAISS vector store creation, persistence, reload, and document deletion
- Mocked LCEL RAG generation & verified source citation extraction
=============================================================================
"""

import os
import io
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from pypdf import PdfWriter
from langchain_core.documents import Document

import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rag
from app import app


@pytest.fixture
def client():
    """Create a Flask test client with testing mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def temp_dirs(monkeypatch):
    """Create isolated temporary directories for vectorstore and uploads during tests."""
    temp_dir = tempfile.mkdtemp()
    test_vectorstore = os.path.join(temp_dir, "vectorstore")
    test_uploads = os.path.join(temp_dir, "uploads")
    os.makedirs(test_vectorstore, exist_ok=True)
    os.makedirs(test_uploads, exist_ok=True)

    # Monkeypatch paths in rag module and app
    monkeypatch.setattr(rag, "VECTORSTORE_DIR", test_vectorstore)
    monkeypatch.setattr(rag, "REGISTRY_FILE", os.path.join(test_vectorstore, "registry.json"))
    monkeypatch.setattr(rag, "CHUNKS_BACKUP_FILE", os.path.join(test_vectorstore, "document_chunks.pkl"))
    app.config["UPLOAD_FOLDER"] = test_uploads

    yield {
        "root": temp_dir,
        "vectorstore": test_vectorstore,
        "uploads": test_uploads
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def create_minimal_pdf_bytes(text: str) -> bytes:
    """Generate a minimal valid PDF byte string containing the specified text."""
    pdf_template = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(text) + 35} >> stream
BT
/F1 12 Tf
72 712 Td
({text}) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000330 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
420
%%EOF"""
    return pdf_template.encode("latin1")


# =============================================================================
# Test 1: Flask Homepage
# =============================================================================
def test_homepage(client):
    """Verify that the homepage renders successfully with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"ResearchMate AI" in response.data
    assert b"Upload Papers" in response.data


# =============================================================================
# Test 2: Missing Question Handling
# =============================================================================
def test_missing_question_handling(client):
    """Verify that missing or empty questions return HTTP 400 Bad Request."""
    # Empty JSON body
    res1 = client.post("/chat", json={})
    assert res1.status_code == 400
    assert "required" in res1.get_json()["error"]

    # Whitespace question
    res2 = client.post("/chat", json={"question": "   "})
    assert res2.status_code == 400
    assert "required" in res2.get_json()["error"]


# =============================================================================
# Test 3: Missing API Key Handling
# =============================================================================
def test_missing_api_key_handling(monkeypatch, temp_dirs):
    """Verify that an unset GROQ_API_KEY produces an informative message instead of a crash."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    # Index a dummy chunk so retriever is active
    dummy_chunk = Document(
        page_content="Deep learning representations in vision transformers.",
        metadata={"document_name": "vit.pdf", "page": 0, "display_page": 1}
    )
    rag.create_or_update_vectorstore(
        chunks=[dummy_chunk],
        document_id="doc_test_1",
        document_name="vit.pdf",
        page_count=1
    )

    response = rag.ask_question("What are vision transformers?")
    assert "GROQ_API_KEY" in response["answer"]
    assert response["sources"] == []


# =============================================================================
# Test 4: Invalid File Upload
# =============================================================================
def test_invalid_file_upload(client):
    """Verify that non-PDF files are rejected."""
    data = {
        "files": (io.BytesIO(b"not a real pdf content"), "malicious_script.txt")
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "only PDF files are allowed" in response.get_json()["error"]


# =============================================================================
# Test 5: Empty Document Behavior & Scanned PDF OCR Detection
# =============================================================================
def test_empty_document_ocr_detection(temp_dirs):
    """Verify that a PDF with zero extractable text triggers an actionable OCR message."""
    empty_pdf_path = os.path.join(temp_dirs["uploads"], "empty_scanned.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(empty_pdf_path, "wb") as f:
        writer.write(f)

    with pytest.raises(ValueError) as excinfo:
        rag.load_pdf(file_path=empty_pdf_path, document_name="empty_scanned.pdf")

    assert "contains no extractable text" in str(excinfo.value)
    assert "OCR" in str(excinfo.value)


# =============================================================================
# Test 6: Chunk Metadata & Page Number Conversion (0-based to 1-based)
# =============================================================================
def test_chunk_metadata_and_page_number_conversion(temp_dirs):
    """
    Verify:
    1. Zero-based page (0) is converted to human-friendly 1-based display_page (1).
    2. Character-based chunking preserves document_name, display_page, and chunk_id.
    """
    raw_text = (
        "Transformer architecture relies entirely on attention mechanisms to draw global "
        "dependencies between input and output without recurrence or convolution. "
        "The Attention Is All You Need paper introduced multi-head self-attention."
    )
    test_pdf_path = os.path.join(temp_dirs["uploads"], "attention.pdf")
    with open(test_pdf_path, "wb") as f:
        f.write(create_minimal_pdf_bytes(raw_text))

    # Test load_pdf
    loaded_docs = rag.load_pdf(test_pdf_path, "attention.pdf")
    assert len(loaded_docs) == 1
    doc = loaded_docs[0]
    assert doc.metadata["document_name"] == "attention.pdf"
    assert doc.metadata["page"] == 0
    assert doc.metadata["display_page"] == 1

    # Test split_documents
    chunks = rag.split_documents(loaded_docs, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["document_name"] == "attention.pdf"
        assert chunk.metadata["display_page"] == 1
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) <= 80


# =============================================================================
# Test 7: Vector Store Persistence, Retrieval, and Clean Deletion
# =============================================================================
def test_vectorstore_persistence_and_deletion(temp_dirs):
    """
    Verify:
    1. Documents are embedded and saved to disk.
    2. Retriever finds relevant chunks with configurable top-k.
    3. Document deletion rebuilds the index cleanly.
    """
    chunk1 = Document(
        page_content="Quantum computing exploits superposition and entanglement.",
        metadata={"document_name": "quantum.pdf", "page": 0, "display_page": 1}
    )
    chunk2 = Document(
        page_content="CRISPR Cas9 enables targeted genome editing in living organisms.",
        metadata={"document_name": "crispr.pdf", "page": 0, "display_page": 1}
    )

    # Ingest document 1
    rag.create_or_update_vectorstore(
        chunks=[chunk1],
        document_id="doc_q1",
        document_name="quantum.pdf",
        page_count=1
    )

    # Ingest document 2
    rag.create_or_update_vectorstore(
        chunks=[chunk2],
        document_id="doc_c1",
        document_name="crispr.pdf",
        page_count=1
    )

    # Verify registry
    docs = rag.list_indexed_documents()
    assert len(docs) == 2

    # Verify retrieval
    retriever = rag.get_retriever(top_k=1)
    results = retriever.invoke("What is quantum entanglement?")
    assert len(results) == 1
    assert "quantum" in results[0].page_content.lower()

    # Delete quantum document
    success = rag.delete_document_from_vectorstore("doc_q1")
    assert success is True

    # Check updated registry
    docs_after = rag.list_indexed_documents()
    assert len(docs_after) == 1
    assert docs_after[0]["id"] == "doc_c1"

    # Verify remaining index only has CRISPR
    new_retriever = rag.get_retriever(top_k=2)
    new_results = new_retriever.invoke("Tell me about genome editing")
    assert len(new_results) == 1
    assert "crispr" in new_results[0].page_content.lower()


# =============================================================================
# Test 8: Mocked RAG Generation & Verified Source Formatting
# =============================================================================
def test_mocked_rag_generation_and_sources(monkeypatch, temp_dirs):
    """Verify end-to-end question answering formatting with a mocked Groq chain."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key_for_testing")

    sample_chunk = Document(
        page_content="ResNet introduces residual skip connections to mitigate vanishing gradients.",
        metadata={"document_name": "resnet.pdf", "page": 2, "display_page": 3}
    )
    rag.create_or_update_vectorstore(
        chunks=[sample_chunk],
        document_id="doc_res1",
        document_name="resnet.pdf",
        page_count=5
    )

    # Mock the LCEL chain invoke output
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = (
        "ResNet uses residual skip connections to allow training of very deep networks [Source 1]."
    )

    with patch("rag.create_rag_chain", return_value=mock_chain):
        result = rag.ask_question("How does ResNet solve vanishing gradients?")

        assert "residual skip connections" in result["answer"]
        assert len(result["sources"]) == 1
        source = result["sources"][0]
        assert source["id"] == 1
        assert source["document"] == "resnet.pdf"
        assert source["page"] == 3
        assert "residual skip connections" in source["text"]


# =============================================================================
# Test 9: Clear Chat & Document Listing Routes
# =============================================================================
def test_routes_clear_chat_and_documents(client, temp_dirs):
    """Test the /clear-chat and /documents endpoints."""
    # Test GET /documents initially empty
    res = client.get("/documents")
    assert res.status_code == 200
    assert res.get_json()["count"] == 0

    # Test POST /clear-chat
    res_clear = client.post("/clear-chat")
    assert res_clear.status_code == 200
    assert "cleared" in res_clear.get_json()["message"]
