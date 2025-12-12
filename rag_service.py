# optimized_rag_service.py
import uuid
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
import math
import torch

from app.config import (
    RAG_INDEX_FILE, 
    RAG_META_FILE, 
    RAG_EMBEDDING_MODEL, 
    RAG_EMBEDDING_DIM,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TOP_K
)

# Limit threads on CPU (tune to your machine)
OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(OMP_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(OMP_THREADS))
torch.set_num_threads(OMP_THREADS)

# Globals
embed_model = None
MODEL_LOCK = threading.Lock()

# We'll maintain:
# - rag_index: FAISS index (IndexFlatIP) wrapped with IndexIDMap
# - rag_meta: dict mapping uuid -> {"text":..., "int_id": ...}
# - next_int_id: incremental integer used for faiss ids (int64)
rag_index = None
rag_meta = {}
next_int_id = 1
DIM = RAG_EMBEDDING_DIM  # fallback; may be updated on model load

def get_embed_model():
    """Lazy-load the embedding model on first use (on CPU)."""
    global embed_model, DIM
    with MODEL_LOCK:
        if embed_model is None:
            try:
                embed_model = SentenceTransformer(RAG_EMBEDDING_MODEL, device="cpu")
                DIM = embed_model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
                raise
    return embed_model

def _ensure_index_initialized():
    """Initialize rag_index (IndexFlatIP + IDMap) with correct DIM."""
    global rag_index, DIM, next_int_id
    if rag_index is not None:
        return
    # If index file exists, try memory-mapped read for large indexes
    if RAG_INDEX_FILE.exists():
        try:
            rag_index = faiss.read_index(str(RAG_INDEX_FILE), faiss.IO_FLAG_MMAP)
        except Exception:
            rag_index = faiss.read_index(str(RAG_INDEX_FILE))
    else:
        # Create a new IndexFlatIP with the configured DIM
        index = faiss.IndexFlatIP(int(DIM))
        # Wrap with ID map so we can associate Faiss ints with our uuids
        rag_index = faiss.IndexIDMap(index)

    # Load metadata if exists
    if RAG_META_FILE.exists():
        try:
            meta = json.loads(RAG_META_FILE.read_text())
            # meta expected to store: {uuid: {"text": ..., "int_id": N}, ... , "_next_int_id": M}
            global rag_meta
            rag_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
            if "_next_int_id" in meta:
                next_int_id = int(meta["_next_int_id"])
        except Exception:
            # If meta can't be read, just keep defaults
            rag_meta = {}
    else:
        rag_meta = {}

def save_rag_state():
    """Persist FAISS index and metadata. Avoid frequent writes by letting caller choose."""
    global rag_index, rag_meta, next_int_id
    if rag_index is None:
        return
    # write index
    faiss.write_index(rag_index, str(RAG_INDEX_FILE))
    # write meta (including next_int_id)
    meta_dump = dict(rag_meta)
    meta_dump["_next_int_id"] = next_int_id
    RAG_META_FILE.write_text(json.dumps(meta_dump, indent=2))

def _normalize_np(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows of a float32 numpy array safely."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr.astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms

def rag_add(text: str, persist: bool = True):
    """
    Add a document to RAG.
    persist: if True, write index+meta to disk after adding. Set False if batching many adds.
    Returns the doc uuid.
    """
    global rag_meta, rag_index, next_int_id

    model = get_embed_model()
    _ensure_index_initialized()

    # compute embedding as numpy float32
    emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    emb = _normalize_np(emb)  # shape (1, DIM)

    int_id = next_int_id
    next_int_id += 1

    # add to faiss (IndexIDMap accepts add_with_ids)
    ids = np.array([int(int_id)], dtype="int64")
    rag_index.add_with_ids(emb, ids)

    # store metadata
    doc_id = str(uuid.uuid4())
    rag_meta[doc_id] = {"text": text, "int_id": int_id}

    if persist:
        save_rag_state()

    return doc_id

def rag_remove(doc_id: str, persist: bool = True):
    """
    Remove a document by uuid.
    Faiss IndexFlat doesn't support delete-in-place; we rebuild index from remaining embeddings.
    """
    global rag_meta, rag_index, next_int_id
    if doc_id not in rag_meta:
        return False

    # remove from metadata
    del rag_meta[doc_id]

    # rebuild index from remaining embeddings saved in metadata (to avoid re-encoding texts,
    # we could optionally persist embeddings separately — for now we re-encode)
    model = get_embed_model()
    # If many docs, batch encode
    remaining_items = list(rag_meta.items())
    if not remaining_items:
        # reset index
        rag_index = faiss.IndexIDMap(faiss.IndexFlatIP(int(DIM)))
        if persist:
            save_rag_state()
        return True

    texts = [v["text"] for _, v in remaining_items]
    int_ids = np.array([v["int_id"] for _, v in remaining_items], dtype="int64")
    embs = []
    batch = 64
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        e = model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
        embs.append(e)
    embs = np.vstack(embs).astype("float32")
    embs = _normalize_np(embs)

    # create fresh index and add with ids
    index = faiss.IndexFlatIP(int(DIM))
    new_idx = faiss.IndexIDMap(index)
    new_idx.add_with_ids(embs, int_ids)

    rag_index = new_idx

    if persist:
        save_rag_state()
    return True

def rag_retrieve(query: str, top_k: int = None, similarity_threshold: float = None):
    """
    Retrieve top-k relevant documents for query.
    Returns list of texts (or empty list).
    """
    if top_k is None:
        top_k = RAG_TOP_K
    if similarity_threshold is None:
        similarity_threshold = RAG_SIMILARITY_THRESHOLD

    _ensure_index_initialized()
    if len(rag_meta) == 0:
        return []

    model = get_embed_model()
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_emb = _normalize_np(q_emb)

    # Search
    D, I = rag_index.search(q_emb, top_k)
    results = []
    # I are faiss int ids or -1 for empty; to map back to uuid we consult rag_meta
    # Build a map int_id -> uuid for fast lookup
    int_to_uuid = {v["int_id"]: k for k, v in rag_meta.items()}

    for sim, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        if sim >= similarity_threshold:
            uuid_key = int_to_uuid.get(int(idx))
            if uuid_key:
                results.append({"id": uuid_key, "text": rag_meta[uuid_key]["text"], "score": float(sim)})
    return results

def rag_list():
    return [{"id": doc_id, "text": v["text"]} for doc_id, v in rag_meta.items()]

def rag_clear(persist: bool = True):
    global rag_index, rag_meta, next_int_id
    rag_index = faiss.IndexIDMap(faiss.IndexFlatIP(int(DIM)))
    rag_meta = {}
    next_int_id = 1
    if persist:
        save_rag_state()
