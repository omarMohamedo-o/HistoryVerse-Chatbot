"""Document ingestion endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from historyverse.api.dependencies import get_document_ingestor
from historyverse.core.schemas import IngestRequest, IngestResponse
from historyverse.services.rag.ingest import DocumentIngestor

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    req: IngestRequest,
    ingestor: DocumentIngestor = Depends(get_document_ingestor),
) -> IngestResponse:
    """Ingest raw documents from disk into the persistent ChromaDB index."""
    logger.info("Starting ingestion from directory: %s", req.source_directory)
    return ingestor.ingest(req.source_directory)
