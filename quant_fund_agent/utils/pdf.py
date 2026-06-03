"""PDF text-extraction helpers used by the Factor Researcher Agent.

The agent feeds raw paper text into an LLM, so we just need a robust,
dependency-light extractor.  We use ``pypdf`` (pure-Python, MIT-licensed)
and stitch page text together with a hard character cap so we never
blow up the LLM context window.

Design notes
------------
- Academic PDFs can be 30+ pages.  Feeding all of it to the LLM would
  be slow and burn tokens for marginal benefit (most of the alpha
  intuition is in the abstract / introduction / methodology).  We
  truncate at ``max_chars`` (default 30 000 chars ≈ 7-8 k tokens).
- Decoding errors on a single page should never crash the agent —
  we swallow them and move on.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("utils.pdf")


def extract_text(
    path: str | Path,
    max_chars: int | None = 30_000,
) -> str:
    """Extract plain text from a PDF, capped at ``max_chars`` characters.

    Args:
        path: path to the PDF file.
        max_chars: maximum number of characters to return; ``None`` for
                   no cap.  We stop reading pages as soon as the cap is
                   hit, so this also bounds runtime.

    Returns:
        Concatenated page text (best-effort), or an empty string if the
        file is unreadable.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency hint
        raise ImportError(
            "pypdf is required for PDF extraction.  "
            "Install with `pip install pypdf`."
        ) from e

    p = Path(path)
    if not p.exists():
        log.warning("PDF not found: %s", p)
        return ""

    try:
        reader = PdfReader(str(p))
    except Exception as e:
        log.warning("Failed to open %s: %s", p, e)
        return ""

    return _reader_to_text(reader, max_chars)


def extract_text_from_bytes(
    data: bytes,
    max_chars: int | None = 30_000,
) -> str:
    """Extract plain text from in-memory PDF ``data`` (e.g. a download).

    Same behaviour as :func:`extract_text` but reads from a byte string
    instead of a file on disk, so a paper fetched over HTTP never has to be
    written out before its text is extracted.  Returns ``""`` if the bytes
    are not a parseable PDF.
    """
    import io

    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency hint
        raise ImportError(
            "pypdf is required for PDF extraction.  "
            "Install with `pip install pypdf`."
        ) from e

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        log.warning("Failed to parse PDF bytes (%d bytes): %s", len(data), e)
        return ""

    return _reader_to_text(reader, max_chars)


def _reader_to_text(reader, max_chars: int | None) -> str:
    """Concatenate page text from a ``pypdf`` reader, capped at ``max_chars``."""
    chunks: list[str] = []
    total = 0
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            log.debug("page %d failed: %s", i, e)
            continue
        chunks.append(text)
        total += len(text)
        if max_chars is not None and total >= max_chars:
            break

    full = "\n\n".join(chunks)
    if max_chars is not None and len(full) > max_chars:
        full = full[:max_chars]
    return full
