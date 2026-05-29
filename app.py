"""
=============================================================================
ResearchMate AI — Flask Web Application (app.py)
=============================================================================
Routes:
- GET  /                : Main user interface
- POST /upload          : Upload and index one or multiple PDF papers
- POST /chat            : Question answering with retrieval & verified citations
- POST /summarize       : Generate structured 8-part paper summary
- GET  /documents       : List all currently indexed documents
- POST /delete-document : Remove paper and rebuild FAISS index
- POST /clear-chat      : Clear session conversation history

Follows security best practices:
- Safe filenames (secure_filename + unique hash)
- Extension & file size validation
- No internal stack traces or API keys leaked to client
- Single-user local session management
=============================================================================
"""

import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, session

import rag

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ResearchMateAI")

# Initialize Flask application
app = Flask(__name__)

# Secret key for session signing
app.secret_key = os.getenv("FLASK_SECRET_KEY", "researchmate-local-dev-secret-key-3.12")

# Upload configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE_MB = 32

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024  # 32 MB max request

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Validate that the uploaded file has a genuine .pdf extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# Routes
# =============================================================================

@app.route("/")
def index():
    """Render the ResearchMate AI main dashboard."""
    return render_template("index.html")


@app.route("/documents", methods=["GET"])
def get_documents():
    """Return all currently indexed research papers with their metadata."""
    try:
        documents = rag.list_indexed_documents()
        return jsonify({"documents": documents, "count": len(documents)})
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return jsonify({"error": "Failed to retrieve documents."}), 500


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Handle single or multiple PDF uploads.
    Extracts text, chunks, embeds, and updates FAISS vector store.
    """
    if "files" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    uploaded_files = request.files.getlist("files")
    if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
        return jsonify({"error": "No file selected for upload."}), 400

    indexed_count = 0
    errors = []

    # Get already indexed documents to prevent duplicate uploads
    existing_docs = {d["filename"]: d["id"] for d in rag.list_indexed_documents()}

    for file in uploaded_files:
        original_name = file.filename
        if not original_name:
            continue

        if not allowed_file(original_name):
            errors.append(f"'{original_name}' rejected: only PDF files are allowed.")
            continue

        # Prevent duplicate upload by filename
        if original_name in existing_docs:
            errors.append(f"'{original_name}' is already indexed in the knowledge base.")
            continue

        # Generate safe filename and unique document ID
        clean_name = secure_filename(original_name)
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        unique_saved_name = f"{doc_id}_{clean_name}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_saved_name)

        try:
            file.save(saved_path)

            # 1. Load PDF pages
            pages = rag.load_pdf(file_path=saved_path, document_name=original_name)

            # 2. Split into character chunks
            chunks = rag.split_documents(pages)

            if not chunks:
                errors.append(f"'{original_name}' produced no usable text chunks.")
                if os.path.exists(saved_path):
                    os.remove(saved_path)
                continue

            # 3. Create or update FAISS index
            rag.create_or_update_vectorstore(
                chunks=chunks,
                document_id=doc_id,
                document_name=original_name,
                page_count=len(pages),
                file_path=saved_path
            )

            indexed_count += 1
            logger.info(f"Successfully indexed '{original_name}' ({len(pages)} pages, {len(chunks)} chunks).")

        except Exception as e:
            logger.error(f"Failed to process '{original_name}': {e}")
            errors.append(f"'{original_name}': {str(e)}")
            # Cleanup on failure
            if os.path.exists(saved_path):
                os.remove(saved_path)

    if indexed_count == 0 and errors:
        return jsonify({"error": "; ".join(errors)}), 400

    response_msg = f"Successfully processed and indexed {indexed_count} paper(s)."
    if errors:
        response_msg += f" (Warnings: {'; '.join(errors)})"

    return jsonify({
        "message": response_msg,
        "indexed_count": indexed_count,
        "warnings": errors
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Answer research questions using RAG.
    Retrieves evidence passages from FAISS and generates an answer via Groq.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question parameter is required and cannot be empty."}), 400

    # Retrieve session chat history (stored as a list of dicts: [{'role': 'user'|'assistant', 'content': '...'}])
    chat_history = session.get("chat_history", [])

    try:
        # Execute RAG query
        result = rag.ask_question(question=question, chat_history=chat_history)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Update session chat history (bounded to last 10 messages)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        session["chat_history"] = chat_history

        return jsonify({
            "question": question,
            "answer": answer,
            "sources": sources
        }), 200

    except Exception as e:
        logger.error(f"Error during chat handling: {e}")
        return jsonify({"error": f"An error occurred while answering your question: {str(e)}"}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    """Generate an 8-part structured summary for a specific uploaded document."""
    data = request.get_json(silent=True) or {}
    document_id = data.get("document_id", "").strip()

    if not document_id:
        return jsonify({"error": "document_id parameter is required."}), 400

    try:
        summary_result = rag.summarize_paper(document_id=document_id)
        return jsonify(summary_result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500


@app.route("/delete-document", methods=["POST"])
def delete_document():
    """Remove a document from the registry and rebuild the FAISS vector index."""
    data = request.get_json(silent=True) or {}
    document_id = data.get("document_id", "").strip()

    if not document_id:
        return jsonify({"error": "document_id parameter is required."}), 400

    try:
        # Lookup file path before removing from registry
        registry = rag._load_registry()
        doc_info = registry.get("documents", {}).get(document_id, {})
        file_path = doc_info.get("file_path", "")

        success = rag.delete_document_from_vectorstore(document_id)
        if not success:
            return jsonify({"error": f"Document '{document_id}' not found."}), 404

        # Remove physical file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as fe:
                logger.warning(f"Could not delete physical file {file_path}: {fe}")

        return jsonify({"message": f"Document '{document_id}' removed successfully."}), 200

    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500


@app.route("/clear-chat", methods=["POST"])
def clear_chat():
    """Clear conversational history from the current user session."""
    session.pop("chat_history", None)
    return jsonify({"message": "Chat history cleared successfully."}), 200


# =============================================================================
# Application Entry Point
# =============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"Starting ResearchMate AI on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
