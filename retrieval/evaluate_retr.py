import json
import os
import pickle
import re
import time
import urllib.request

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


# ============================================================
# PATHS
# ============================================================

EVALUATION_QUERIES_PATH = "retrieval/eval_queries.json"

BM25_INDEX_PATH = "retrieval/bm25_index.pkl"

EMBEDDING_INDEX_PATH = "retrieval/embedding_index.faiss"
METADATA_PATH = "retrieval/metadata.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Toggle off to reproduce pre-graph-expansion retrieval behavior exactly
ENABLE_GRAPH_EXPANSION = True

# BGE embedding models expect an instruction prefix on the query side only
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Weight on normalized dense score vs. normalized reranker score in the final blend
RERANK_FUSION_ALPHA = 0.9

reranker = CrossEncoder(RERANKER_MODEL)


# ============================================================
# LOAD DATA
# ============================================================

def load_evaluation_queries():

    with open(
        EVALUATION_QUERIES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_bm25():

    with open(
        BM25_INDEX_PATH,
        "rb"
    ) as f:

        data = pickle.load(f)

    return data["bm25"], data["docs"]


def load_bge():

    index = faiss.read_index(
        EMBEDDING_INDEX_PATH
    )

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
# SCORE NORMALIZATION
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
# BM25
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

    return [
        {
            "index": int(i),
            "id": docs[i].get("id"),
            "title": docs[i].get("title"),
            "score": float(scores[i])
        }
        for i in indices
    ]


# ============================================================
# BGE
# ============================================================

def search_bge(
    query,
    index,
    docs,
    model,
    top_k=5
):

    query_embedding = model.encode(
        [BGE_QUERY_INSTRUCTION + query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    scores = scores[0]
    indices = indices[0]

    return [
        {
            "index": int(i),
            "id": docs[i].get("id"),
            "title": docs[i].get("title"),
            "text": docs[i].get("text", ""),
            "score": float(score),
            "metadata": docs[i].get("metadata", {})
        }
        for i, score in zip(indices, scores)
    ]


# ============================================================
# RERANK TEXT CLEANING
# ============================================================

RERANK_TEXT_MAX_CHARS = 800

HEADER_LINE_RE = re.compile(r"^#{1,6}\s+")
SKIP_SECTION_HEADERS = ("#### Visual References", "## Provenance")


def build_rerank_text(raw_text, max_chars=RERANK_TEXT_MAX_CHARS):
    """Strip markdown headers/image/provenance boilerplate, keeping only concept content."""

    if not raw_text:
        return ""

    kept_lines = []
    skipping = False

    for line in raw_text.splitlines():

        stripped = line.strip()

        if HEADER_LINE_RE.match(stripped):
            skipping = stripped.startswith(SKIP_SECTION_HEADERS)
            continue

        if skipping:
            continue

        if stripped.startswith("- Image:") or stripped.startswith("- File:"):
            continue

        if stripped:
            kept_lines.append(stripped)

    cleaned = re.sub(r"\s+", " ", " ".join(kept_lines)).strip()

    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(" ")

    return truncated[:last_space] if last_space > 0 else truncated


def clean_context_text(text: str) -> str:
    """
    Deterministically clean OKF concept text before sending it
    to the local LLM.

    This removes structural/provenance boilerplate but preserves
    actual knowledge, procedures, tables, warnings, notes, and
    visual references.
    """

    if not text:
        return ""

    lines = text.splitlines()
    cleaned = []

    # Structural headings that do not add useful information
    removable_headings = {
        "### Concept",
        "### Note",
        "### Procedure",
        "### Warning",
        "### Specification",
        "### Component",
        "### Maintenance",
        "### Troubleshooting",
        "### Table",
        "## Knowledge",
        "## Tables",
        "## Provenance",
    }

    # Lines/sections that are purely metadata/provenance
    skip_provenance = False

    for line in lines:
        stripped = line.strip()

        # Empty lines
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        # Remove YAML/frontmatter if it is present in the text
        if stripped == "---":
            continue

        # Remove frontmatter-style metadata
        if any(
            stripped.startswith(prefix)
            for prefix in [
                "id:",
                "type:",
                "title:",
                "description:",
                "tags:",
                "sources:",
                "document_id:",
                "source_file:",
                "page_start:",
                "page_end:",
                "source_block:",
            ]
        ):
            continue

        # Skip provenance section completely
        if stripped == "## Provenance":
            skip_provenance = True
            continue

        if skip_provenance:
            # Stop skipping if another major section starts
            if stripped.startswith("## ") and stripped != "## Provenance":
                skip_provenance = False
            else:
                continue

        # Remove purely structural headings
        if stripped in removable_headings:
            continue

        # Remove duplicate top-level title if it appears
        # as both "# TITLE" and frontmatter title.
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                cleaned.append(title)
            continue

        cleaned.append(line)

    # Collapse excessive blank lines
    result = []
    previous_blank = False

    for line in cleaned:
        if line.strip() == "":
            if not previous_blank:
                result.append("")
            previous_blank = True
        else:
            result.append(line.strip())
            previous_blank = False

    return "\n".join(result).strip()


def clean_retrieved_context(results):
    """
    Clean the text of retrieved/reranked results while preserving
    retrieval metadata such as ID, title, score and OKF metadata.
    """

    cleaned_results = []

    for result in results:
        cleaned_text = clean_context_text(
            result.get("text", "")
        )

        cleaned_result = dict(result)
        cleaned_result["text"] = cleaned_text

        cleaned_results.append(cleaned_result)

    return cleaned_results


# ============================================================
# CONTEXT ASSEMBLY
# ============================================================

def build_context(results, max_results=5):
    """
    Convert retrieved OKF concepts into LLM-ready context.

    Preserves:
    - concept ID
    - title
    - knowledge type
    - source document
    - source pages
    - knowledge content
    - tables
    - visual references

    Removes the duplicated ## Provenance section from the
    raw OKF text because provenance is represented explicitly.
    """

    if not results:
        return ""

    context_parts = []

    for rank, result in enumerate(results[:max_results], start=1):

        concept_id = result.get("id", "")
        title = result.get("title", "")
        text = result.get("text", "")
        metadata = result.get("metadata") or {}

        if not text:
            continue

        # ----------------------------------------------------
        # Remove YAML/frontmatter if it is present in the text
        # ----------------------------------------------------

        content = text.strip()

        if content.startswith("---"):
            parts = content.split("---", 2)

            if len(parts) == 3:
                content = parts[2].strip()

        # ----------------------------------------------------
        # Remove ## Provenance from the content.
        # Provenance is added explicitly below.
        # ----------------------------------------------------

        content = re.split(
            r"^##\s+Provenance\s*$",
            content,
            maxsplit=1,
            flags=re.MULTILINE
        )[0].strip()

        # ----------------------------------------------------
        # Extract source information
        # ----------------------------------------------------

        sources = metadata.get("sources") or []

        source_file = None
        page_start = None
        page_end = None

        if sources:
            first_source = sources[0]

            source_file = first_source.get("source_file")
            page_start = first_source.get("page_start")
            page_end = first_source.get("page_end")

        # ----------------------------------------------------
        # Build source block
        # ----------------------------------------------------

        source_lines = []

        if source_file:
            source_lines.append(
                f"Source file: {source_file}"
            )

        if page_start is not None:

            if page_end is not None and page_end != page_start:
                source_lines.append(
                    f"Pages: {page_start}-{page_end}"
                )
            else:
                source_lines.append(
                    f"Page: {page_start}"
                )

        # ----------------------------------------------------
        # Build retrieved concept block
        # ----------------------------------------------------

        block = [
            f"[SOURCE {rank}]",
            f"Concept ID: {concept_id}",
            f"Title: {title}",
            f"Type: {metadata.get('type', 'concept')}",
        ]

        if source_lines:
            block.extend(source_lines)

        block.append("")
        block.append(content)

        context_parts.append(
            "\n".join(block)
        )

    return "\n\n".join(context_parts)


# ============================================================
# CONTEXTUAL COMPRESSION (LOCAL GEMMA VIA LITERT-LM)
# ============================================================
# Reuses the same local litert-lm chat-completions server/model as
# backend/server.py's translate_with_gemma() — no HF load, no new process.

LITERT_LM_CHAT_URL = "http://localhost:9379/v1/chat/completions"
GEMMA_MODEL_NAME = "gemma4-e2b"
GEMMA_COMPRESSION_TIMEOUT = 240
GEMMA_COMPRESSION_MAX_TOKENS = 1024

# Gemma (via litert-lm) rejects requests over this many input tokens outright
GEMMA_MAX_INPUT_TOKENS = 4096
# Safety margin under the hard limit, to absorb token-estimate error + prompt overhead
GEMMA_PROMPT_TOKEN_BUDGET = 3000
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_token_count(text):
    """Conservative, tokenizer-free token estimate (~4 chars/token for English)."""
    return (len(text) // CHARS_PER_TOKEN_ESTIMATE) + 1


def split_into_sections(text):
    """Split cleaned concept text into paragraph-level sections, dropping Visual Reference noise."""

    sections = []
    current = []

    for line in text.splitlines():

        stripped = line.strip()

        if stripped == "#### Visual References":
            continue

        if stripped.startswith("- Image:") or stripped.startswith("- File:"):
            continue

        if not stripped:
            if current:
                sections.append(" ".join(current).strip())
                current = []
            continue

        current.append(stripped)

    if current:
        sections.append(" ".join(current).strip())

    return [section for section in sections if section]


def score_section_relevance(query, section):
    """Cheap lexical-overlap relevance score in [0, 1] — no extra model call."""

    query_words = set(re.findall(r"\w+", query.lower()))
    section_words = set(re.findall(r"\w+", section.lower()))

    if not query_words or not section_words:
        return 0.0

    return len(query_words & section_words) / len(query_words)


def build_source_header(rank, result):
    """Build the '[SOURCE N] ...' provenance header for one retrieved result."""

    metadata = result.get("metadata") or {}
    sources = metadata.get("sources") or []

    header_lines = [
        f"[SOURCE {rank}]",
        f"Concept ID: {result.get('id', '')}",
        f"Title: {result.get('title', '')}",
        f"Type: {metadata.get('type', 'concept')}",
    ]

    if sources:
        first_source = sources[0]
        source_file = first_source.get("source_file")
        page_start = first_source.get("page_start")
        page_end = first_source.get("page_end")

        if source_file:
            header_lines.append(f"Source file: {source_file}")

        if page_start is not None:
            if page_end is not None and page_end != page_start:
                header_lines.append(f"Pages: {page_start}-{page_end}")
            else:
                header_lines.append(f"Page: {page_start}")

    return "\n".join(header_lines)


def select_relevant_context_for_compression(
    query,
    results,
    max_results=5,
    token_budget=GEMMA_PROMPT_TOKEN_BUDGET
):
    """
    Build a query-aware, token-budgeted context for the Gemma compression call:
    keep only the most query-relevant sections per source (most relevant first),
    stopping once the token budget is reached. Source/provenance headers stay
    attached to whichever sections survive for that source; sources with no
    relevant sections are dropped entirely.
    """

    used_tokens = 0
    blocks = []

    for rank, result in enumerate(results[:max_results], start=1):

        header = build_source_header(rank, result)
        sections = split_into_sections(result.get("text", ""))

        scored_sections = sorted(
            (
                (score_section_relevance(query, section), section)
                for section in sections
            ),
            key=lambda pair: pair[0],
            reverse=True
        )

        header_tokens = estimate_token_count(header)
        source_tokens = header_tokens
        kept_sections = []

        for score, section in scored_sections:

            if score <= 0.0:
                break

            section_tokens = estimate_token_count(section)

            if used_tokens + source_tokens + section_tokens > token_budget:
                break

            kept_sections.append(section)
            source_tokens += section_tokens

        if not kept_sections:
            continue

        used_tokens += source_tokens
        blocks.append(header + "\n\n" + "\n\n".join(kept_sections))

        if used_tokens >= token_budget:
            break

    return "\n\n---\n\n".join(blocks)


COMPRESSION_PROMPT_TEMPLATE = """You are a context compression assistant for a retrieval-augmented system.

User query:
{query}

Below is retrieved context made up of [SOURCE N] blocks, each tagged with a
Concept ID, Title, and page/source reference.

Instructions:
- Keep ONLY evidence relevant to answering the user query.
- Preserve exact facts, numbers, procedures, warnings, and table contents verbatim.
- Keep each [SOURCE N] block's Concept ID / Title / page tag attached to its compressed content.
- Do NOT invent information that is not present in the context.
- Do NOT answer or summarize the user's question itself.
- If a source block has no relevant evidence, omit that block entirely.

Retrieved context:
{context}
"""


def compress_context_with_gemma(query, context, results=None, timeout=GEMMA_COMPRESSION_TIMEOUT):
    """
    Query-aware contextual compression via the local Gemma model served by
    litert-lm. The context sent to Gemma is first reduced to only the most
    query-relevant sections of `results`, budgeted to stay under Gemma's
    4096-token input limit. Falls back to the original Python-cleaned
    `context` if the reduced context still doesn't fit, or if the LLM call
    fails or times out for any reason.
    """

    if not context:
        return context

    reduced_context = (
        select_relevant_context_for_compression(query, results)
        if results
        else context
    )

    if not reduced_context:
        return context

    prompt = COMPRESSION_PROMPT_TEMPLATE.format(query=query, context=reduced_context)

    if estimate_token_count(prompt) > GEMMA_MAX_INPUT_TOKENS:
        print("[WARN] Reduced context still exceeds Gemma's input token limit; falling back to Python-cleaned context.")
        return context

    payload = {
        "model": GEMMA_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": GEMMA_COMPRESSION_MAX_TOKENS,
        "temperature": 0.0,
    }

    print(
        f"[DEBUG] Gemma compression request | "
        f"prompt_chars={len(prompt)} "
        f"estimated_tokens={estimate_token_count(prompt)} "
        f"output_token_limit={GEMMA_COMPRESSION_MAX_TOKENS}"
    )

    request = urllib.request.Request(
        LITERT_LM_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.monotonic()
    print(f"[DEBUG] Gemma compression request start_time={start_time:.3f}")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

        end_time = time.monotonic()
        print(
            f"[DEBUG] Gemma compression request end_time={end_time:.3f} "
            f"elapsed={end_time - start_time:.3f}s"
        )

        compressed = result["choices"][0]["message"]["content"].strip()

        if not compressed:
            raise ValueError("Empty compression response from Gemma")

        return compressed

    except Exception as exc:
        end_time = time.monotonic()
        print(
            f"[DEBUG] Gemma compression request end_time={end_time:.3f} "
            f"elapsed={end_time - start_time:.3f}s (failed)"
        )
        print(f"[WARN] Gemma contextual compression failed ({exc}); falling back to Python-cleaned context.")
        return context

def rerank_results(query, results, top_k=5, alpha=RERANK_FUSION_ALPHA):
    if not results:
        return []

    pairs = [
        (query, build_rerank_text(result.get("text", "")))
        for result in results
    ]

    rerank_scores = np.asarray(reranker.predict(pairs), dtype="float32")

    dense_scores = np.array(
        [result.get("score", 0.0) for result in results],
        dtype="float32"
    )

    normalized_dense = normalize_scores(dense_scores)
    normalized_rerank = normalize_scores(rerank_scores)

    reranked = []

    for result, rerank_score, norm_dense, norm_rerank in zip(
        results, rerank_scores, normalized_dense, normalized_rerank
    ):
        item = result.copy()
        item["reranker_score"] = float(rerank_score)
        item["final_score"] = float(
            alpha * norm_dense + (1 - alpha) * norm_rerank
        )
        reranked.append(item)

    reranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return reranked[:top_k]


# ============================================================
# ONE-HOP GRAPH-LINK EXPANSION
# ============================================================

RELATED_CONCEPTS_HEADER_RE = re.compile(r"^##\s+Related Concepts\s*$", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def extract_related_concept_links(text):
    """Return link targets listed under a '## Related Concepts' section, if present."""

    if not text:
        return []

    header_match = RELATED_CONCEPTS_HEADER_RE.search(text)

    if not header_match:
        return []

    section_start = header_match.end()

    next_header_match = re.search(r"^##\s+\S", text[section_start:], re.MULTILINE)

    section_end = (
        section_start + next_header_match.start()
        if next_header_match
        else len(text)
    )

    section_text = text[section_start:section_end]

    return [
        target.strip()
        for _, target in MARKDOWN_LINK_RE.findall(section_text)
    ]


def build_link_resolution_index(docs):
    """Map concept file basenames and ids to their position in `docs`."""

    index_by_basename = {}
    index_by_id = {}

    for doc_index, doc in enumerate(docs):

        path = doc.get("path")

        if path:
            index_by_basename[os.path.basename(path)] = doc_index

        doc_id = doc.get("id")

        if doc_id:
            index_by_id[doc_id] = doc_index

    return index_by_basename, index_by_id


def resolve_related_concept_link(link_target, index_by_basename, index_by_id):
    """Resolve one '## Related Concepts' link to a doc index already in the KB, or None."""

    cleaned = link_target.split("#")[0].strip()

    if not cleaned:
        return None

    basename = os.path.basename(cleaned)

    if basename in index_by_basename:
        return index_by_basename[basename]

    stem = os.path.splitext(basename)[0]

    if stem in index_by_id:
        return index_by_id[stem]

    return None


def expand_with_related_concepts(candidates, docs):
    """
    One-hop graph-link expansion.

    For each candidate's concept file, parse its '## Related Concepts' links,
    resolve them to concept files already in the knowledge base, and add any
    not already in the candidate set.
    """

    index_by_basename, index_by_id = build_link_resolution_index(docs)

    expanded = list(candidates)
    seen_indices = {result["index"] for result in candidates}

    for result in candidates:

        doc = docs[result["index"]]

        link_targets = extract_related_concept_links(doc.get("text", ""))

        for link_target in link_targets:

            linked_index = resolve_related_concept_link(
                link_target,
                index_by_basename,
                index_by_id
            )

            if linked_index is None or linked_index in seen_indices:
                continue

            seen_indices.add(linked_index)

            linked_doc = docs[linked_index]

            expanded.append(
                {
                    "index": linked_index,
                    "id": linked_doc.get("id"),
                    "title": linked_doc.get("title"),
                    "text": linked_doc.get("text", ""),
                    "score": result.get("score", 0.0),
                    "source": "graph_expansion"
                }
            )

    return expanded


def search_bge_reranked(
    query,
    embedding_index,
    bge_docs,
    model,
    candidate_k=20,
    top_k=5,
    enable_graph_expansion=ENABLE_GRAPH_EXPANSION
):
    candidates = search_bge(query, index=embedding_index, docs=bge_docs, model=model, top_k=candidate_k)

    # initial retrieval -> graph expansion -> rerank -> final top-N
    if enable_graph_expansion:
        candidates = expand_with_related_concepts(candidates, bge_docs)

    results = rerank_results(
        query,
        candidates,
        top_k=top_k
    )

    return clean_retrieved_context(results)


# ============================================================
# CANONICAL RETRIEVAL PIPELINE
# ============================================================
# This is the production retrieval strategy: dense search -> graph expansion
# -> rerank -> top-N. BM25 / BGE-only below are kept only as evaluation
# baselines for comparison.

def retrieve(
    query,
    embedding_index,
    bge_docs,
    model,
    top_k=5,
    candidate_k=20,
    enable_graph_expansion=ENABLE_GRAPH_EXPANSION
):
    return search_bge_reranked(
        query,
        embedding_index,
        bge_docs,
        model,
        candidate_k=candidate_k,
        top_k=top_k,
        enable_graph_expansion=enable_graph_expansion
    )


# ============================================================
# RECALL
# ============================================================

def recall_at_k(
    results,
    expected_ids,
    k
):

    retrieved_ids = {
        result["id"]
        for result in results[:k]
    }

    return int(
        bool(
            retrieved_ids.intersection(
                expected_ids
            )
        )
    )


# ============================================================
# EVALUATE ONE METHOD
# ============================================================

def evaluate_method(
    queries,
    search_function,
    method_name
):

    recall_1 = 0
    recall_3 = 0
    recall_5 = 0

    print()
    print("=" * 70)
    print(method_name)
    print("=" * 70)

    for item in queries:

        query_id = item["id"]
        query = item["query"]

        expected_ids = set(
            item["expected_concepts"]
        )

        results = search_function(
            query
        )

        r1 = recall_at_k(
            results,
            expected_ids,
            1
        )

        r3 = recall_at_k(
            results,
            expected_ids,
            3
        )

        r5 = recall_at_k(
            results,
            expected_ids,
            5
        )

        recall_1 += r1
        recall_3 += r3
        recall_5 += r5

        print(
            f"{query_id:>4} | "
            f"R@1={r1} "
            f"R@3={r3} "
            f"R@5={r5} | "
            f"{query}"
        )

    total = len(queries)

    print()
    print(
        f"{method_name} Recall@1: "
        f"{recall_1}/{total} = "
        f"{recall_1 / total:.3f}"
    )

    print(
        f"{method_name} Recall@3: "
        f"{recall_3}/{total} = "
        f"{recall_3 / total:.3f}"
    )

    print(
        f"{method_name} Recall@5: "
        f"{recall_5}/{total} = "
        f"{recall_5 / total:.3f}"
    )

    return {
        "recall@1": recall_1 / total,
        "recall@3": recall_3 / total,
        "recall@5": recall_5 / total
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading evaluation queries...")

    queries = load_evaluation_queries()

    print(
        "Evaluation queries:",
        len(queries)
    )


    # ========================================================
    # Load BM25
    # ========================================================

    print()
    print("Loading BM25 index...")

    bm25, bm25_docs = load_bm25()


    # ========================================================
    # Load BGE
    # ========================================================

    print()
    print("Loading BGE index...")

    embedding_index, bge_docs, model = load_bge()


    # ========================================================
    # BM25 evaluation
    # ========================================================

    bm25_results = evaluate_method(

        queries,

        lambda query:
            search_bm25(
                query,
                bm25,
                bm25_docs,
                top_k=5
            ),

        "BM25"
    )


    # ========================================================
    # BGE evaluation
    # ========================================================

    bge_results = evaluate_method(

        queries,

        lambda query:
            search_bge(
                query,
                embedding_index,
                bge_docs,
                model,
                top_k=5
            ),

        "BGE"
    )

    # ========================================================
    # PRODUCTION PIPELINE EVALUATION (DENSE + RERANK)
    # ========================================================

    pipeline_results = evaluate_method(
        queries,
        lambda query:
            retrieve(
                query,
                embedding_index,
                bge_docs,
                model,
                candidate_k=20,
                top_k=5
            ),
        "PRODUCTION PIPELINE (DENSE + RERANK)"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 80)

    print()

    print(
        f"{'BM25':<30}"
        f"{bm25_results['recall@1']:<15.3f}"
        f"{bm25_results['recall@3']:<15.3f}"
        f"{bm25_results['recall@5']:<15.3f}"
    )


    # ----------------------------
    # BGE
    # ----------------------------

    print(
        f"{'BGE':<30}"
        f"{bge_results['recall@1']:<15.3f}"
        f"{bge_results['recall@3']:<15.3f}"
        f"{bge_results['recall@5']:<15.3f}"
    )


    # ----------------------------
    # PRODUCTION PIPELINE (DENSE + GRAPH EXPANSION + RERANK)
    # ----------------------------

    print(
        f"{'Production Pipeline':<30}"
        f"{pipeline_results['recall@1']:<15.3f}"
        f"{pipeline_results['recall@3']:<15.3f}"
        f"{pipeline_results['recall@5']:<15.3f}"
    )

    # ========================================================
    # CONTEXT ASSEMBLY TEST
    # ========================================================

    test_query = "How do I pair my phone using Bluetooth?"

    print()
    print("=" * 80)
    print("CONTEXT ASSEMBLY TEST")
    print("=" * 80)
    print()
    print("Query:")
    print(test_query)
    print()

    retrieved = retrieve(
        test_query,
        embedding_index,
        bge_docs,
        model,
        candidate_k=20,
        top_k=5,
        enable_graph_expansion=False
    )

    context = build_context(
        retrieved,
        max_results=5
    )

    print("RETRIEVED CONTEXT")
    print("=" * 80)
    print(context)
    print("=" * 80)

    compressed_context = compress_context_with_gemma(test_query, context, results=retrieved)

    print()
    print("COMPRESSED CONTEXT (GEMMA)")
    print("=" * 80)
    print(compressed_context)
    print("=" * 80)


if __name__ == "__main__":
    main()