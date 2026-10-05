import os
import json
import yaml
import pickle

import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


OKF_PATH = "okf_bundle/concepts"

BM25_INDEX_PATH = "retrieval/bm25_index.pkl"
EMBEDDING_INDEX_PATH = "retrieval/embedding_index.faiss"
METADATA_PATH = "retrieval/metadata.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_okf_documents():

    documents = []

    for root, dirs, files in os.walk(OKF_PATH):

        for file in files:

            if not file.endswith(".md"):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split YAML frontmatter
            if content.startswith("---"):

                parts = content.split("---", 2)

                yaml_text = parts[1]

                # Remove invalid control characters
                yaml_text = "".join(
                    char
                    for char in yaml_text
                    if char in "\n\r\t" or ord(char) >= 32
                )

                metadata = yaml.safe_load(yaml_text) or {}

                text = parts[2]

            else:

                metadata = {}
                text = content

            documents.append(
                {
                    "id": metadata.get("id"),
                    "title": metadata.get("title"),
                    "text": text,
                    "metadata": metadata,
                    "path": path
                }
            )

    return documents


def build_bm25_index(docs):

    corpus = []

    for doc in docs:

        title = doc.get("title") or ""
        text = doc.get("text") or ""

        combined_text = title + " " + text

        corpus.append(
            combined_text.lower().split()
        )

    bm25 = BM25Okapi(corpus)

    with open(
        BM25_INDEX_PATH,
        "wb"
    ) as f:

        pickle.dump(
            {
                "bm25": bm25,
                "docs": docs
            },
            f
        )

    print("BM25 index created")


def build_embedding_index(docs):

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = []

    for doc in docs:

        title = doc.get("title") or ""
        text = doc.get("text") or ""

        texts.append(
            title
            + "\n"
            + text
        )

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        EMBEDDING_INDEX_PATH
    )

    print("FAISS index created")
    print("Embedding dimension:", dimension)
    print("FAISS vectors:", index.ntotal)


def save_metadata(docs):

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            docs,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Metadata saved")


def main():

    print("Loading OKF concepts...")

    docs = load_okf_documents()

    print(
        "Documents loaded:",
        len(docs)
    )

    if not docs:

        print(
            "ERROR: No OKF concept files found."
        )

        return

    # Phase 2/3: BM25
    build_bm25_index(docs)

    # Phase 4/5: BGE + FAISS
    build_embedding_index(docs)

    # Save mapping between vector/BM25 position and OKF concept
    save_metadata(docs)

    print()
    print("======================================")
    print("RETRIEVAL INDEX BUILD COMPLETE")
    print("======================================")
    print("Documents:", len(docs))


if __name__ == "__main__":
    main()