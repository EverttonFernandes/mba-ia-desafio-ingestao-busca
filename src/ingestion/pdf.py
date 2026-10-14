"""Leitura e particionamento do PDF, metadata e identidade dos chunks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .errors import IngestError

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def prepare_pdf(pdf_path: Path) -> tuple[list[Document], str]:
    """Preserva pagina zero-based e origem; nao envia o documento a rede."""
    pages, pdf_hash = _load_pdf(pdf_path)
    text_pages = _select_text_pages(pages)
    _set_page_metadata(text_pages, pdf_path.name, pdf_hash)
    chunks = _split_pages(text_pages)
    return chunks, pdf_hash


def _load_pdf(pdf_path: Path) -> tuple[list[Document], str]:
    if not pdf_path.is_file():
        raise IngestError("PDF ausente. Coloque document.pdf na raiz ou use --pdf CAMINHO.")
    try:
        pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        pages = PyPDFLoader(str(pdf_path)).load()
    except Exception:
        raise IngestError(
            "Nao foi possivel ler o PDF. Confira se e valido, legivel "
            "e nao protegido por senha."
        ) from None
    return pages, pdf_hash


def _select_text_pages(pages: list[Document]) -> list[Document]:
    text_pages = [page for page in pages if page.page_content.strip()]
    if not text_pages:
        raise IngestError(
            "PDF vazio ou sem texto extraivel. Use um PDF textual; "
            "OCR nao esta incluido."
        )
    return text_pages


def _set_page_metadata(pages: list[Document], source: str, pdf_hash: str) -> None:
    # Guarda somente metadata util; nao persiste caminhos absolutos da maquina.
    for page in pages:
        page.metadata = {
            "source": source,
            "page": page.metadata["page"],
            "pdf_sha256": pdf_hash,
        }


def _split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def collection_metadata(
    settings: Settings, chunks: list[Document], pdf_hash: str,
) -> dict:
    """Identifica documento, modelo e particionamento para evitar misturar versoes."""
    # Detecta tambem mudancas no loader/splitter, mesmo com PDF identico.
    chunks_payload = [[chunk.page_content, chunk.metadata] for chunk in chunks]
    chunks_hash = hashlib.sha256(
        json.dumps(chunks_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "ingestion_version": 1,
        "pdf_sha256": pdf_hash,
        "chunks_sha256": chunks_hash,
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
    }


def chunk_ids(
    collection_name: str, collection_metadata: dict, chunks: list[Document]
) -> list[str]:
    """Os mesmos dados produzem os mesmos IDs, inclusive depois de reiniciar."""
    identity_prefix = collection_name + json.dumps(collection_metadata, sort_keys=True)
    return [
        hashlib.sha256(f"{identity_prefix}:{index}".encode()).hexdigest()
        for index in range(len(chunks))
    ]
