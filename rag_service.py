# rag_service.py
"""
Persistent RAG service (FAISS + SQLite) with stable vid mapping (IndexIDMap).
Functions provided (keep these names so existing routes work):
- rag_add(text, source="manual", metadata=None, lang="unknown") -> dict
- rag_remove(vid) -> bool
- rag_retrieve(query, top_k=5) -> dict
- rag_list(limit=50, offset=0) -> dict
- rag_clear() -> None
"""

import os
import sqlite3
import json
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configurable via env vars
RAG_STORE = os.getenv("RAG_STORE", "rag_store")
META_DB = os.path.join(RAG_STORE, "rag_meta.sqlite3")
INDEX_FILE = os.path.join(RAG_STORE, "faiss.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "384"))  # should match model dim; will adapt if model differs
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

_lock = threading.Lock()

# Lazy-loaded objects
_embed_model: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.Index] = None  # will be IndexIDMap(IndexFlatIP(dim))


# ----------------------
# Utilities
# ----------------------
def ensure_store_dirs():
    os.makedirs(RAG_STORE, exist_ok=True)


def get_embedding_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def normalize_vectors(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    txt = text.replace("\n", " ").strip()
    if len(txt) <= chunk_size:
        return [txt]
    chunks = []
    start = 0
    L = len(txt)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(txt[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def init_sqlite():
    ensure_store_dirs()
    conn = sqlite3.connect(META_DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            vid INTEGER PRIMARY KEY,
            source TEXT,
            original_text TEXT,
            english_text TEXT,
            lang TEXT,
            metadata TEXT,
            created_at REAL
        );
        """
    )
    conn.commit()
    return conn


# ----------------------
# FAISS index helpers
# ----------------------
def _create_faiss_index(dim: int) -> faiss.Index:
    # Using IndexFlatIP (inner product) with normalized vectors -> cosine similarity
    base = faiss.IndexFlatIP(dim)
    idmap = faiss.IndexIDMap(base)
    return idmap


def load_faiss_index(dim: int) -> faiss.Index:
    global _faiss_index
    if _faiss_index is not None:
        # if index dimension mismatch, rebuild
        try:
            if hasattr(_faiss_index, "d") and _faiss_index.d != dim:
                _faiss_index = _create_faiss_index(dim)
        except Exception:
            _faiss_index = _create_faiss_index(dim)
        return _faiss_index

    if os.path.exists(INDEX_FILE):
        try:
            idx = faiss.read_index(INDEX_FILE)
            _faiss_index = idx
            return _faiss_index
        except Exception:
            # corrupt or incompatible index -> recreate
            _faiss_index = _create_faiss_index(dim)
            return _faiss_index
    else:
        _faiss_index = _create_faiss_index(dim)
        return _faiss_index


def save_faiss_index():
    global _faiss_index
    if _faiss_index is None:
        return
    faiss.write_index(_faiss_index, INDEX_FILE)


def rebuild_index_from_db(conn: sqlite3.Connection, embed_func):
    """
    Rebuild FAISS index by re-embedding all english_text rows and adding with their vid as id.
    embed_func: function(list_of_texts) -> np.ndarray (n,dim) normalized float32
    """
    global _faiss_index
    cur = conn.cursor()
    cur.execute("SELECT vid, english_text FROM chunks ORDER BY vid ASC")
    rows = cur.fetchall()
    if not rows:
        # create empty index
        _faiss_index = _create_faiss_index(EMBED_DIM)
        save_faiss_index()
        return

    vids = [int(r[0]) for r in rows]
    texts = [r[1] for r in rows]
    embs = embed_func(texts)  # normalized np.float32
    # faiss requires int64 ids
    ids = np.array(vids, dtype=np.int64)
    dim = embs.shape[1]
    _faiss_index = _create_faiss_index(dim)
    _faiss_index.add_with_ids(embs, ids)
    save_faiss_index()


# ----------------------
# DB helpers
# ----------------------
def get_next_vid(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("SELECT MAX(vid) FROM chunks")
    r = cur.fetchone()
    if r is None or r[0] is None:
        return 0
    return int(r[0]) + 1


def insert_meta_rows(conn: sqlite3.Connection, rows: List[Tuple[int, str, str, str, str, str, float]]):
    """
    rows: list of tuples (vid, source, original_text, english_text, lang, metadata_json, created_at)
    """
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO chunks (vid, source, original_text, english_text, lang, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def delete_meta_vid(conn: sqlite3.Connection, vid: int) -> bool:
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE vid = ?", (vid,))
    changed = cur.rowcount
    conn.commit()
    return changed > 0


def fetch_meta_by_vids(conn: sqlite3.Connection, vids: List[int]) -> List[Dict[str, Any]]:
    if not vids:
        return []
    q = ",".join("?" for _ in vids)
    cur = conn.cursor()
    cur.execute(f"SELECT vid, source, original_text, english_text, lang, metadata, created_at FROM chunks WHERE vid IN ({q})", vids)
    rows = cur.fetchall()
    out = []
    # Map rows by vid for order preservation
    row_map = {int(r[0]): r for r in rows}
    for v in vids:
        r = row_map.get(int(v))
        if not r:
            continue
        out.append(
            {
                "vid": int(r[0]),
                "source": r[1],
                "original_text": r[2],
                "english_text": r[3],
                "lang": r[4],
                "metadata": json.loads(r[5]) if r[5] else {},
                "created_at": float(r[6]) if r[6] else None,
            }
        )
    return out


def list_meta(conn: sqlite3.Connection, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    total = int(cur.fetchone()[0])
    cur.execute("SELECT vid, source, original_text, english_text, lang, metadata, created_at FROM chunks ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = cur.fetchall()
    items = []
    for r in rows:
        items.append(
            {
                "vid": int(r[0]),
                "source": r[1],
                "original_text": r[2],
                "english_text": r[3],
                "lang": r[4],
                "metadata": json.loads(r[5]) if r[5] else {},
                "created_at": float(r[6]) if r[6] else None,
            }
        )
    return {"total": total, "items": items}


# ----------------------
# Embedding helper
# ----------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns normalized float32 embeddings shaped (n, dim).
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    model = get_embedding_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    embs = embs.astype(np.float32)
    embs = normalize_vectors(embs)
    return embs


# ----------------------
# Public functions (API surface)
# ----------------------
# Initialize DB & index on import
_conn_global = init_sqlite()
# Try to load the model to get correct dim (but do lazily to avoid heavy startup)
try:
    _model = get_embedding_model()
    EMBED_DIM = _model.get_sentence_embedding_dimension() or EMBED_DIM
except Exception:
    # if model fails to load now, we'll try later when needed
    pass

# Ensure FAISS index object exists & consistent dim
_faiss_index = load_faiss_index(EMBED_DIM)


def rag_add(text: str, source: str = "manual", metadata: Optional[Dict] = None, lang: str = "unknown") -> Dict[str, Any]:
    """
    Add text (possibly long) to the RAG store. Returns dict with added counts and vid range.
    - chunking performed, embeddings computed in batch
    - uses explicit vid ids via IndexIDMap
    """
    if not text or not text.strip():
        return {"added": 0, "message": "empty text"}

    with _lock:
        # chunk (english_text == text here; translation should occur before calling if needed)
        original_text = text.strip()
        english_chunks = chunk_text(original_text)
        if not english_chunks:
            return {"added": 0}

        # compute embeddings in one batch
        embs = embed_texts(english_chunks)
        dim = embs.shape[1]
        # reload index if dim changed
        global _faiss_index
        if _faiss_index is None or (_faiss_index.d != dim if hasattr(_faiss_index, "d") else False):
            _faiss_index = load_faiss_index(dim)

        # compute vids to assign
        start_vid = get_next_vid(_conn_global)
        vids = np.array([start_vid + i for i in range(len(english_chunks))], dtype=np.int64)

        # add with ids
        _faiss_index.add_with_ids(embs, vids)
        save_faiss_index()

        # write metadata rows
        now = time.time()
        rows = []
        for i, chunk in enumerate(english_chunks):
            vid = int(start_vid + i)
            rows.append((vid, source, original_text, chunk, lang, json.dumps(metadata or {}), now))
        insert_meta_rows(_conn_global, rows)

        return {"added": len(english_chunks), "start_vid": int(start_vid), "end_vid_exclusive": int(start_vid + len(english_chunks))}


def rag_remove(vid: int) -> bool:
    """
    Remove a chunk by vid. We remove metadata and rebuild the index from DB to keep mapping stable.
    Returns True if removed, False otherwise.
    """
    with _lock:
        removed = delete_meta_vid(_conn_global, int(vid))
        if not removed:
            return False
        # rebuild index from remaining rows (re-embedding)
        rebuild_index_from_db(_conn_global, embed_texts)
        return True


def rag_retrieve(query: str, top_k: int = 5, min_score: Optional[float] = None) -> Dict[str, Any]:
    """
    Retrieve top_k results for query.
    Returns dict: { "query": ..., "results": [ {vid, source, english_text, original_text, lang, metadata, created_at, score}, ... ] }
    score is cosine similarity in [-1,1] (for normalized vectors inner product).
    """
    if not query or not query.strip():
        return {"query": query, "results": []}

    with _lock:
        em = embed_texts([query])
        if em.shape[0] == 0:
            return {"query": query, "results": []}

        # ensure index is loaded
        idx = load_faiss_index(em.shape[1])
        if idx.ntotal == 0:
            return {"query": query, "results": []}

        D, I = idx.search(em, top_k)
        scores = D[0].tolist()
        ids = I[0].tolist()
        # filter -1 indices
        pairs = [(int(i), float(s)) for i, s in zip(ids, scores) if int(i) != -1]
        # optional min_score filter
        if min_score is not None:
            pairs = [(i, s) for i, s in pairs if s >= float(min_score)]
        if not pairs:
            return {"query": query, "results": []}

        vids = [p[0] for p in pairs]
        vid_to_score = {p[0]: p[1] for p in pairs}
        rows = fetch_meta_by_vids(_conn_global, vids)
        # attach scores in same order as vids
        results = []
        for r in rows:
            r["score"] = float(vid_to_score.get(r["vid"], 0.0))
            results.append(r)
        return {"query": query, "results": results}


def rag_list(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    with _lock:
        return list_meta(_conn_global, limit=limit, offset=offset)


def rag_clear() -> None:
    """
    Clear the entire store: drop DB table rows and delete index file.
    """
    with _lock:
        cur = _conn_global.cursor()
        cur.execute("DELETE FROM chunks")
        _conn_global.commit()
        # remove index file on disk and recreate empty index
        try:
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
        except Exception:
            pass
        # reset in-memory index
        global _faiss_index
        _faiss_index = load_faiss_index(EMBED_DIM)
        save_faiss_index()


# ----------------------
# If imported, ensure index is consistent with DB
# ----------------------
def _ensure_index_consistency():
    """
    On import, ensure FAISS index contains entries matching DB. If index exists but is empty
    or missing some vids, rebuild from DB (safe).
    """
    with _lock:
        cur = _conn_global.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        total = int(cur.fetchone()[0])
        idx = load_faiss_index(EMBED_DIM)
        if total == 0 and idx.ntotal == 0:
            return
        # If counts differ, rebuild to be safe
        if total != idx.ntotal:
            rebuild_index_from_db(_conn_global, embed_texts)


# run consistency check at import
try:
    _ensure_index_consistency()
except Exception:
    # don't crash on import - will rebuild on first operation
    pass

# For convenience: small test when run directly (manual)
if __name__ == "__main__":
    print("RAG service module. Available functions: rag_add, rag_remove, rag_retrieve, rag_list, rag_clear")
    # simple smoke test (won't run until model/faiss available)
    # example usage:
    # print(rag_add("The President of India as of 2025 is Ram Singh"))
    pass
