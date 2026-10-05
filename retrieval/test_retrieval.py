import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BM25_INDEX_PATH = "retrieval/bm25_index.pkl"

EMBEDDING_INDEX_PATH = "retrieval/embedding_index.faiss"

METADATA_PATH = "retrieval/metadata.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# LOAD BM25
# ============================================================

def load_bm25():

    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["docs"]


# ============================================================
# LOAD BGE
# ============================================================

def load_bge():

    index = faiss.read_index(
        EMBEDDING_INDEX_PATH
    )

    import json

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        docs = json.load(f)

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return index, docs, model


# ============================================================
# NORMALIZE BM25 SCORES
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype="float32"
    )

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.ones_like(scores)

    return (
        (scores - min_score)
        /
        (max_score - min_score)
    )


# ============================================================
# BM25 SEARCH
# ============================================================

def search_bm25(
    query,
    bm25,
    docs,
    top_k=5
):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(
        query_tokens
    )

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in indices:

        doc = docs[index]

        results.append(
            {
                "index": int(index),
                "id": doc.get("id"),
                "title": doc.get("title"),
                "score": float(scores[index]),
                "path": doc.get("path")
            }
        )

    return results


# ============================================================
# BGE SEARCH
# ============================================================

def search_bge(
    query,
    embedding_index,
    docs,
    model,
    top_k=5
):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = embedding_index.search(
        query_embedding,
        top_k
    )

    scores = scores[0]
    indices = indices[0]

    results = []

    for index, score in zip(
        indices,
        scores
    ):

        doc = docs[index]

        results.append(
            {
                "index": int(index),
                "id": doc.get("id"),
                "title": doc.get("title"),
                "score": float(score),
                "path": doc.get("path")
            }
        )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def search_hybrid(
    query,
    bm25,
    bm25_docs,
    embedding_index,
    bge_docs,
    model,
    top_k=5,
    candidate_k=20,
    bm25_weight=0.5
):

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:candidate_k]


    # --------------------------------------------------------
    # BGE
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    bge_scores, bge_indices = embedding_index.search(
        query_embedding,
        candidate_k
    )

    bge_scores = bge_scores[0]
    bge_indices = bge_indices[0]


    # --------------------------------------------------------
    # Candidate union
    # --------------------------------------------------------

    candidate_ids = set(
        int(i)
        for i in bm25_indices
    )

    candidate_ids.update(
        int(i)
        for i in bge_indices
    )


    # --------------------------------------------------------
    # Normalize BM25 scores
    # --------------------------------------------------------

    bm25_candidate_scores = np.array(
        [
            bm25_scores[i]
            for i in candidate_ids
        ],
        dtype="float32"
    )

    normalized_bm25 = normalize_scores(
        bm25_candidate_scores
    )

    bm25_normalized_map = {
        doc_index: float(score)
        for doc_index, score in zip(
            candidate_ids,
            normalized_bm25
        )
    }


    # --------------------------------------------------------
    # BGE score map
    # --------------------------------------------------------

    bge_score_map = {
        int(index): float(score)
        for index, score in zip(
            bge_indices,
            bge_scores
        )
    }


    # --------------------------------------------------------
    # Hybrid scoring
    # --------------------------------------------------------

    semantic_weight = (
        1.0 - bm25_weight
    )

    results = []

    for doc_index in candidate_ids:

        lexical_score = (
            bm25_normalized_map.get(
                doc_index,
                0.0
            )
        )

        semantic_score = (
            bge_score_map.get(
                doc_index,
                0.0
            )
        )

        final_score = (
            bm25_weight * lexical_score
            +
            semantic_weight * semantic_score
        )

        doc = bm25_docs[doc_index]

        results.append(
            {
                "index": int(doc_index),
                "id": doc.get("id"),
                "title": doc.get("title"),
                "score": float(final_score),
                "bm25_score": float(
                    lexical_score
                ),
                "bge_score": float(
                    semantic_score
                ),
                "path": doc.get("path")
            }
        )


    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
    method
):

    print()
    print(
        f"----- {method} -----"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n#{rank}"
        )

        print(
            f"ID      : "
            f"{result['id']}"
        )

        print(
            f"Title   : "
            f"{result['title']}"
        )

        print(
            f"Score   : "
            f"{result['score']:.4f}"
        )

        if method == "HYBRID":

            print(
                f"BM25    : "
                f"{result['bm25_score']:.4f}"
            )

            print(
                f"BGE     : "
                f"{result['bge_score']:.4f}"
            )

        print(
            f"Path    : "
            f"{result['path']}"
        )


# ============================================================
# RUN MANUAL QUERY
# ============================================================

def run_manual_query(
    query,
    bm25,
    bm25_docs,
    embedding_index,
    bge_docs,
    model
):

    print()
    print("=" * 80)
    print("QUERY:", query)
    print("=" * 80)


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = search_bm25(
        query,
        bm25,
        bm25_docs,
        top_k=5
    )

    print_results(
        bm25_results,
        "BM25"
    )


    # --------------------------------------------------------
    # BGE
    # --------------------------------------------------------

    bge_results = search_bge(
        query,
        embedding_index,
        bge_docs,
        model,
        top_k=5
    )

    print_results(
        bge_results,
        "BGE"
    )


    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_results = search_hybrid(
        query,
        bm25,
        bm25_docs,
        embedding_index,
        bge_docs,
        model,
        top_k=5,
        candidate_k=20,
        bm25_weight=0.5
    )

    print_results(
        hybrid_results,
        "HYBRID"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HARrier RETRIEVAL TEST")
    print("=" * 80)


    # --------------------------------------------------------
    # Load BM25
    # --------------------------------------------------------

    print()
    print("Loading BM25 index...")

    bm25, bm25_docs = load_bm25()

    print(
        "BM25 documents:",
        len(bm25_docs)
    )


    # --------------------------------------------------------
    # Load BGE
    # --------------------------------------------------------

    print()
    print("Loading BGE index...")

    embedding_index, bge_docs, model = load_bge()

    print(
        "BGE documents:",
        len(bge_docs)
    )

    print(
        "FAISS vectors:",
        embedding_index.ntotal
    )


    # --------------------------------------------------------
    # Manual query loop
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MANUAL QUERY MODE")
    print("=" * 80)

    print()
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        print()

        query = input(
            "Enter your query: "
        ).strip()

        if not query:
            print(
                "Please enter a query."
            )
            continue

        if query.lower() in {
            "exit",
            "quit"
        }:
            print(
                "Exiting retrieval test."
            )
            break

        run_manual_query(
            query,
            bm25,
            bm25_docs,
            embedding_index,
            bge_docs,
            model
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()